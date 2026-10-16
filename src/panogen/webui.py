"""로컬 웹 UI — 프롬프트(+첨부 최대 3장) → 진행도 → 초안 확인(컨펌/수정) → 이음새 보정 → 고른 시드만 8K.
360° 파노라마와 일반 이미지(--type flat) 둘 다. 일반 이미지는 이음새 단계 없이 확인 → 고해상도.

  panogen ui [--port 8370]      → http://127.0.0.1:8370/

GPU 작업은 `python -m panogen generate|upscale` 하위 프로세스로 한 번에 하나만 돌리고,
그 출력(로그 줄 + diffusers tqdm 진행 막대)을 읽어 진행도를 만든다. 작업이 끝나면 프로세스가
종료되므로 모델 메모리가 매번 반환된다 (16GB 맥에서 서버가 메모리를 붙잡지 않도록).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from panogen.config import Config
from panogen.imagein import ASPECTS

MIN_FREE_GB = float(os.environ.get("PANOGEN_MIN_FREE_GB", 3))      # 이보다 디스크가 적으면 일시정지
RESUME_FREE_GB = float(os.environ.get("PANOGEN_RESUME_FREE_GB", 6))  # 이만큼 비면 자동으로 이어서 (경계에서 멈춤·재개 반복 방지)


def disk_free_gb(path: Path) -> float:
    import shutil
    return shutil.disk_usage(path).free / 1024**3
TQDM = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+)")
IMG_IDX = re.compile(r"\((\d+)/(\d+)\)")
TILE = re.compile(r"타일 (\d+)/(\d+)")
BENCH = re.compile(r"===== bench: (\S+) =====")
GEN_DONE = re.compile(r"\s*생성 \d+s\s*$")


def slug(text: str, n: int = 24) -> str:
    s = re.sub(r"[^\w\-]+", "_", text.strip().lower(), flags=re.UNICODE).strip("_")
    return s[:n] or "pano"


class Job:
    """하위 프로세스 하나의 상태. 출력 줄을 읽으며 phase/step 을 갱신."""

    def __init__(self, kind: str, cmd: list[str], name: str, root: Path, label: str | None = None):
        self.kind, self.cmd, self.name = kind, cmd, name
        self.label = label or kind      # 화면 제목용 (upscale --seam-only → "seam")
        self.state = "running"          # running | paused(디스크 부족) | done | error | cancelled
        self.root = root
        self.reason: str | None = None
        self.free_gb: float | None = None
        self.phase = "starting"         # encoding | loading | generating | seam | postprocess | upscaling
        self.image = [0, 0]             # 몇 번째 이미지 / 전체
        self.step = [0, 0]              # 샘플링 step / 전체
        self.eta = ""                   # tqdm 남은 시간
        self.tile = [0, 0]
        self.sub = ""                   # bench: 지금 돌고 있는 백엔드 폴더
        self.in_seam = False
        self.lines: list[str] = []
        self.started = time.time()
        self.ended: float | None = None
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
        self.proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=0)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._disk_guard, args=(root,), daemon=True).start()

    def _disk_guard(self, root: Path):
        """맥은 메모리가 모자라면 스왑이 디스크를 채운다(20B 모델 1장에 30GB 이상). 0 이 되면 시스템이 멈출 수 있어서
        여유가 MIN_FREE_GB 아래로 떨어지면 프로세스를 일시정지(메모리 요청이 멈춰 스왑도 더 늘지 않음)하고,
        사용자가 공간을 비워 RESUME_FREE_GB 이상이 되면 자동으로 이어서 한다. 진행 상황은 그대로 유지."""
        while self.state in ("running", "paused"):
            self.free_gb = round(disk_free_gb(root), 1)
            if self.state == "running" and self.free_gb < MIN_FREE_GB:
                self.pause(f"디스크 여유 {self.free_gb}GB — {MIN_FREE_GB:g}GB 미만이라 일시정지했습니다. "
                           f"공간을 비우면 {RESUME_FREE_GB:g}GB 이상에서 자동으로 이어집니다")
            elif self.state == "paused" and self.reason == "disk" and self.free_gb >= RESUME_FREE_GB:
                self.resume(f"디스크 여유 {self.free_gb}GB — 이어서 진행합니다")
            time.sleep(2 if self.state == "paused" else 5)

    def _procs(self):
        """생성 프로세스와 그 자식들 (psutil)"""
        import psutil

        try:
            p = psutil.Process(self.proc.pid)
            return [p, *p.children(recursive=True)]
        except psutil.Error:
            return []

    def pause(self, msg: str, reason: str = "disk"):
        if self.state != "running":
            return
        for p in self._procs():
            try:
                p.suspend()
            except Exception:
                pass
        self.state, self.reason, self.paused_at = "paused", reason, time.time()
        self.lines.append(msg)

    def resume(self, msg: str = "이어서 진행합니다"):
        if self.state != "paused":
            return
        for p in reversed(self._procs()):
            try:
                p.resume()
            except Exception:
                pass
        self.paused_total = getattr(self, "paused_total", 0) + time.time() - getattr(self, "paused_at", time.time())
        self.state, self.reason = "running", None
        self.lines.append(msg)

    def _read(self):
        buf = b""
        while chunk := self.proc.stdout.read(256):
            buf += chunk
            *parts, buf = re.split(rb"[\r\n]", buf)
            for p in parts:
                if p.strip():
                    self._line(p.decode("utf-8", "replace"))
        if buf.strip():
            self._line(buf.decode("utf-8", "replace"))
        code = self.proc.wait()
        if self.state == "running":
            self.state = "done" if code == 0 else "error"
        self.ended = time.time()

    def _line(self, s: str):
        is_bar = "|" in s and TQDM.search(s)
        if not is_bar:
            self.lines = (self.lines + [s.rstrip()])[-200:]
        if (m := BENCH.search(s)):
            self.sub, self.image, self.step = m[1], [0, 0], [0, 0]
            self.phase = "starting"
        elif "받는 중" in s:
            self.phase = "downloading"
        elif "[1/3]" in s:
            self.phase = "encoding"
        elif "[2/3]" in s:
            self.phase = "loading"
        elif "준비 완료" in s:   # 이음새 인페인팅 중 모델을 다시 올린 경우는 seam 유지
            self.phase = "seam" if self.in_seam else "generating"
        elif "[3/3]" in s or s.startswith("[upscale]"):
            if (m := IMG_IDX.search(s)):
                self.image = [int(m[1]), int(m[2])]
            self.step, self.tile = [0, 0], [0, 0]
            if s.startswith("[upscale]"):
                self.phase = "upscaling"
            elif self.phase == "postprocess":     # 두 번째 이미지부터는 모델이 이미 올라가 있음
                self.phase = "generating"
        elif GEN_DONE.match(s):
            self.phase = "postprocess"
        elif s.startswith("[seam]"):          # 초안 이음새 보정 (upscale 의 앞단계)
            if (m := IMG_IDX.search(s)):
                self.image = [int(m[1]), int(m[2])]
            self.phase, self.step, self.tile, self.in_seam = "seam", [0, 0], [0, 0], True
        elif "이음새 인페인팅 (" in s:
            self.phase, self.step, self.in_seam = "seam", [0, 0], True
        elif "이음새 인페인팅 완료" in s:
            self.phase, self.in_seam = "postprocess", False
        elif s.lstrip().startswith("디테일 강화:"):
            self.phase, self.tile = "refining", [0, 0]
        elif s.startswith("[refine]"):
            if (m := IMG_IDX.search(s)):
                self.image = [int(m[1]), int(m[2])]
            self.phase, self.tile = "refining", [0, 0]
        elif "업스케일" in s and self.kind == "generate":
            self.phase = "upscaling"
        elif (m := TILE.search(s)):
            self.tile = [int(m[1]), int(m[2])]
        if is_bar and self.phase in ("generating", "seam"):
            m = TQDM.search(s)
            self.step, self.eta = [int(m[1]), int(m[2])], m[4]

    def cancel(self):
        if self.state == "paused":          # 정지된 프로세스는 종료 신호를 처리하지 못하므로 먼저 깨움
            self.resume("중단합니다")
        if self.state == "running":
            self.state = "cancelled"
            self.proc.terminate()

    def to_dict(self) -> dict:
        return {"kind": self.kind, "label": self.label, "name": self.name, "sub": self.sub, "state": self.state, "reason": getattr(self, "reason", None), "phase": self.phase,
                "image": self.image, "step": self.step, "eta": self.eta, "tile": self.tile,
                "elapsed": round((self.ended or time.time()) - self.started), "log": self.lines[-40:],
                "free_gb": self.free_gb, "min_free_gb": MIN_FREE_GB, "resume_free_gb": RESUME_FREE_GB}


def list_results(out_root: Path, name: str) -> list[dict]:
    """결과 목록. 360 초안(이음새 보정 전 *_raw.png)과 보정본(*_2k.png)·일반 이미지(*_2k.png)를 한 줄씩."""
    d = out_root / name
    url = lambda f: f"/outputs/{name}/{f.name}"
    inputs = [url(f) for f in sorted(d.glob("input*.*")) if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    stems = sorted({f.stem.removesuffix("_2k") for f in d.glob("*_2k.png")} |
                   {f.stem.removesuffix("_raw") for f in d.glob("*_raw.png")})
    items = []
    for stem in stems:
        meta_p = d / f"{stem}.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        fixed, raw = d / f"{stem}_2k.png", d / f"{stem}_raw.png"
        big = sorted(d.glob(f"{stem}_*k.jpg")) + sorted(d.glob(f"{stem}_*px.jpg"))
        items.append({"stem": stem, "seed": meta.get("seed"), "prompt": meta.get("prompt", ""),
                      "type": meta.get("type", "pano"), "size": meta.get("size"),
                      "backend": meta.get("backend", "dit360"), "fast": meta.get("fast", False),
                      "refined": meta.get("refined", False),
                      "seam": meta.get("seam_verdict"), "seconds": meta.get("generate_seconds"),
                      "peak_gb": meta.get("peak_memory_gb"), "steps": meta.get("steps"),
                      "seam_pending": not fixed.exists(), "inpainted": bool(meta.get("seam_inpaint")),
                      "img2k": url(fixed if fixed.exists() else raw), "raw": url(raw) if raw.exists() else None,
                      "input": inputs[0] if inputs else None, "inputs": inputs, "input_meta": meta.get("input"),
                      "img8k": url(big[0]) if big else None})
    return items


def list_sessions(out_root: Path) -> list[dict]:
    out = []
    for d in sorted((p for p in out_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        metas = sorted(d.glob("*.json"))
        if not metas:
            continue
        m = json.loads(metas[0].read_text(encoding="utf-8"))
        prompt = m.get("prompt", "")
        stems = {f.stem.removesuffix("_2k") for f in d.glob("*_2k.png")} | {f.stem.removesuffix("_raw") for f in d.glob("*_raw.png")}
        out.append({"name": d.name, "prompt": prompt, "count": len(stems), "type": m.get("type", "pano"),
                    "backend": m.get("backend", "dit360") + (" fast" if m.get("fast") else ""),
                    "mtime": time.strftime("%m-%d %H:%M", time.localtime(d.stat().st_mtime))})
    return out[:30]


def _cached(repo: str, filename: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    r = try_to_load_from_cache(repo, filename)
    return isinstance(r, str) and Path(r).exists()


def _weights_cached(repo: str, sub: str) -> bool:
    """sub 폴더의 가중치(단일 파일 또는 샤드 전부)가 캐시에 있는지 — 네트워크 없이 확인."""
    from huggingface_hub import try_to_load_from_cache

    for idx_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        idx = try_to_load_from_cache(repo, f"{sub}/{idx_name}")
        if isinstance(idx, str) and Path(idx).exists():
            shards = set(json.loads(Path(idx).read_text(encoding="utf-8"))["weight_map"].values())
            return all(_cached(repo, f"{sub}/{f}") for f in shards)
    return any(_cached(repo, f"{sub}/{f}") for f in ("model.safetensors", "diffusion_pytorch_model.safetensors"))


def _repo_cached(repo: str) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def model_status(root: Path, name: str, b: dict, gguf: bool) -> dict:
    """백엔드별 준비 상태 (받은 것 / 없는 것). 다운로드 버튼과 생성 가능 여부 표시에 씀."""
    base = b.get("base", "")
    missing = []
    subs = ["text_encoder", "vae"] + (["text_encoder_2"] if b.get("kind") == "dit360" else [])
    for sub in subs:
        if not _weights_cached(base, sub):
            missing.append(f"{sub}")
    if gguf:
        if not (b.get("gguf_repo") and _cached(b["gguf_repo"], b["gguf_file"])):
            missing.append(f"transformer (GGUF {b.get('gguf_file', '')})")
    elif not _weights_cached(base, "transformer"):
        missing.append("transformer")
    if b.get("lora"):
        ok = _cached(b["lora"], b["lora_file"]) if b.get("lora_file") else _repo_cached(b["lora"])
        if not ok:
            missing.append("파노라마 LoRA")
    if b.get("kind") == "dit360" and not (root / "third_party" / "DiT360" / "src" / "pipeline.py").exists():
        missing.append("DiT360 코드 (scripts/fetch_dit360.py)")
    fast = bool(b.get("speed_lora"))
    fast_ready = fast and _cached(b["speed_lora"], b["speed_lora_file"])
    return {"name": name, "base": base, "kind": b.get("kind"), "ready": not missing, "missing": missing,
            "fast": fast, "fast_ready": fast_ready, "refine": b.get("kind") in ("dit360", "qwen"),
            "gated": base.startswith("black-forest-labs/")}


def bench_summary(out_root: Path, tag: str) -> dict | None:
    p = out_root / f"bench_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def list_benches(out_root: Path) -> list[dict]:
    out = []
    for p in sorted(out_root.glob("bench_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        s = json.loads(p.read_text(encoding="utf-8"))
        out.append({"tag": s["tag"], "created": s.get("created", ""), "prompts": len(s.get("prompts", [])),
                    "seeds": s.get("seeds", []),
                    "runs": [{"label": r["backend"] + (" fast" if r.get("fast") else ""), "status": r["status"]}
                             for r in s.get("runs", [])]})
    return out[:20]


def compare(out_root: Path, tag: str) -> dict | None:
    """비교 표: 행 = 프롬프트×시드, 열 = 백엔드 실행. 칸 = 2K 이미지와 시간·메모리·이음새."""
    s = bench_summary(out_root, tag)
    if not s:
        return None
    pidx = {p: i for i, p in enumerate(s["prompts"])}
    for r in s["runs"]:
        r["label"] = r["backend"] + (" fast" if r.get("fast") else "")
        cells = {}
        if (out_root / r["dir"]).is_dir():
            for it in list_results(out_root, r["dir"]):
                if it["prompt"] in pidx:
                    cells[f"{pidx[it['prompt']]}_{it['seed']}"] = it
        r["cells"] = cells
    return s


def serve(cfg: Config, port: int = 8370, extra_config: Path | None = None) -> None:
    root = cfg.root
    out_root = cfg.get_path(cfg["output"]["dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    page = root / "ui" / "index.html"
    base_cmd = [sys.executable, "-m", "panogen"] + (["--config", str(extra_config)] if extra_config else [])
    state: dict = {"job": None}
    backends = cfg.get("backends", {})
    lock = threading.Lock()
    try:   # 맥(mps-16g)은 트랜스포머를 GGUF 로 받으므로 준비 상태 확인 기준이 다름
        from panogen.device import detect
        from panogen.profiles import resolve

        d = detect()
        gguf = resolve(cfg["generate"]["profile"], d["device"], d["vram_gb"]).transformer == "gguf"
    except Exception:
        gguf = False

    def start(kind: str, args: list[str], name: str, label: str | None = None) -> Job:
        with lock:
            j = state["job"]
            if j and j.state in ("running", "paused"):
                raise RuntimeError("이미 작업이 진행 중입니다 (일시정지 포함).")
            free = disk_free_gb(root)
            if free < RESUME_FREE_GB:
                raise RuntimeError(f"디스크 여유가 {free:.1f}GB 입니다. {RESUME_FREE_GB:g}GB 이상 비운 뒤 시작하세요 "
                                   "(맥은 생성 중 스왑이 디스크를 수십 GB 씁니다).")
            state["job"] = Job(kind, base_cmd + [kind] + args, name, root, label)
            return state["job"]

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, ctype: str):
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _upload(self, n: int):
            import io

            from PIL import Image

            if n <= 0 or n > 40 * 1024**2:
                return self._json({"error": "이미지는 40MB 이하만 첨부할 수 있습니다."}, 400)
            data = self.rfile.read(n)
            try:
                im = Image.open(io.BytesIO(data))
                im.verify()
                im = Image.open(io.BytesIO(data))
            except Exception:
                return self._json({"error": "이미지 파일이 아닙니다."}, 400)
            ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(im.format)
            if not ext:
                return self._json({"error": f"지원하지 않는 형식: {im.format} (JPG·PNG·WEBP)"}, 400)
            up = out_root / "_uploads"
            up.mkdir(exist_ok=True)
            fid = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}{ext}"
            (up / fid).write_bytes(data)
            from panogen.imagein import is_panorama

            return self._json({"id": fid, "url": f"/outputs/_uploads/{fid}", "width": im.width,
                               "height": im.height, "panorama": is_panorama(im.width, im.height)})

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._file(page, "text/html; charset=utf-8")
            if u.path == "/api/job":
                j = state["job"]
                return self._json(j.to_dict() if j else None)
            if u.path == "/api/backends":
                default = cfg.get("model", {}).get("backend", "dit360")
                return self._json([{"name": k, "default": k == default, "fast": bool(v.get("speed_lora")),
                                    "refine": v.get("kind") in ("dit360", "qwen"), "base": v.get("base", "")}
                                   for k, v in backends.items()])
            if u.path == "/api/models":
                import shutil

                hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
                hub.mkdir(parents=True, exist_ok=True)
                free = shutil.disk_usage(hub).free / 1024**3
                return self._json({"disk_free_gb": round(free, 1), "gguf": gguf,
                                   "default": cfg.get("model", {}).get("backend", "dit360"),
                                   "models": [model_status(root, k, v, gguf) for k, v in backends.items()]})
            if u.path == "/api/benches":
                return self._json(list_benches(out_root))
            if u.path == "/api/compare":
                tag = (q.get("tag") or [""])[0]
                c = compare(out_root, tag) if re.fullmatch(r"[\w\-]+", tag or "") else None
                return self._json(c) if c else self._json({"error": "없는 비교"}, 404)
            if u.path == "/api/sessions":
                return self._json(list_sessions(out_root))
            if u.path == "/api/results":
                name = (q.get("name") or [""])[0]
                if not name or "/" in name or ".." in name or not (out_root / name).is_dir():
                    return self._json({"error": "없는 폴더"}, 404)
                return self._json(list_results(out_root, name))
            if u.path.startswith("/outputs/"):
                f = (out_root / unquote(u.path[len("/outputs/"):])).resolve()
                if out_root.resolve() not in f.parents or not f.is_file():
                    return self.send_error(HTTPStatus.NOT_FOUND)
                ctype = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                         ".webp": "image/webp"}.get(f.suffix.lower(), "application/json")
                return self._file(f, ctype)
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            if u.path == "/api/upload":   # 첨부 이미지 (본문 = 이미지 바이트)
                return self._upload(n)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if u.path == "/api/generate":
                    prompt = str(body.get("prompt", "")).strip()
                    if not prompt:
                        raise ValueError("프롬프트를 입력하세요.")
                    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{slug(prompt)}"
                    args = ["-p", prompt, "-n", name, "-c", str(max(1, min(8, int(body.get("count") or 1)))),
                            "--no-upscale"]
                    if body.get("backend"):
                        if body["backend"] not in backends:
                            raise ValueError(f"알 수 없는 모델: {body['backend']}")
                        args += ["--backend", body["backend"]]
                    if body.get("fast"):
                        args.append("--fast")
                    if body.get("seed") not in (None, ""):
                        args += ["--seed", str(int(body["seed"]))]
                    if body.get("steps") not in (None, ""):
                        args += ["--steps", str(max(1, min(60, int(body["steps"]))))]
                    kind = "flat" if body.get("type") == "flat" else "pano"
                    if kind == "flat":
                        aspect = str(body.get("aspect") or "auto")
                        if aspect != "auto" and aspect not in ASPECTS:
                            raise ValueError(f"알 수 없는 비율: {aspect}")
                        args += ["--type", "flat", "--aspect", aspect]
                    else:
                        args.append("--seam-later")      # 이음새 보정 전 초안을 먼저 확인
                    ids = body.get("images") or ([body["image"]] if body.get("image") else [])
                    if len(ids) > 3:
                        raise ValueError("첨부 이미지는 최대 3장입니다.")
                    for fid in map(str, ids):
                        img = out_root / "_uploads" / fid
                        if not re.fullmatch(r"[\w\-]+\.(jpg|png|webp)", fid) or not img.is_file():
                            raise ValueError("첨부 이미지를 다시 올려 주세요.")
                        args += ["--image", str(img)]
                    if ids:
                        if body.get("strength") not in (None, ""):
                            args += ["--strength", str(max(0.0, min(1.0, float(body["strength"]))))]
                        if kind == "pano" and body.get("fov") not in (None, ""):
                            args += ["--fov", str(max(30.0, min(170.0, float(body["fov"]))))]
                    return self._json(start("generate", args, name).to_dict())
                if u.path == "/api/upscale":
                    name = str(body.get("name", ""))
                    seeds = [str(int(s)) for s in body.get("seeds", [])]
                    if not seeds or "/" in name or ".." in name or not (out_root / name).is_dir():
                        raise ValueError("8K 로 만들 시드를 고르세요.")
                    seam_only = bool(body.get("seam_only"))
                    args = [str(out_root / name), "--seed", *seeds] + (["--refine"] if body.get("refine") else []) \
                        + (["--seam-only"] if seam_only else [])
                    return self._json(start("upscale", args, name, "seam" if seam_only else None).to_dict())
                if u.path == "/api/download":
                    be = str(body.get("backend", ""))
                    if be not in backends:
                        raise ValueError(f"알 수 없는 모델: {be}")
                    return self._json(start("download", ["--backend", be] + (["--fast"] if body.get("fast") else []),
                                            be).to_dict())
                if u.path == "/api/bench":
                    runs = []
                    for r in body.get("runs", []):
                        if r.get("backend") not in backends:
                            raise ValueError(f"알 수 없는 모델: {r.get('backend')}")
                        runs.append(r["backend"] + (":fast" if r.get("fast") else ""))
                    seeds = [str(int(x)) for x in body.get("seeds", [])] or ["1"]
                    if not runs:
                        raise ValueError("비교할 모델을 하나 이상 고르세요.")
                    tag = time.strftime("%Y%m%d_%H%M%S")
                    args = ["--backends", ",".join(runs), "--seeds", *seeds, "--tag", tag]
                    if body.get("limit"):
                        args += ["--limit", str(max(1, int(body["limit"])))]
                    return self._json(start("bench", args, tag).to_dict())
                if u.path == "/api/resume":     # 디스크 일시정지에서 바로 이어서 (최소 여유는 넘어야 함)
                    j = state["job"]
                    if not j or j.state != "paused":
                        raise ValueError("일시정지된 작업이 없습니다.")
                    free = disk_free_gb(root)
                    if free < MIN_FREE_GB:
                        raise ValueError(f"디스크 여유가 아직 {free:.1f}GB 입니다. {MIN_FREE_GB:g}GB 이상 비워 주세요.")
                    j.resume(f"디스크 여유 {free:.1f}GB — 직접 이어서 진행합니다")
                    return self._json(j.to_dict())
                if u.path == "/api/cancel":
                    if state["job"]:
                        state["job"].cancel()
                    return self._json({"ok": True})
            except (RuntimeError, ValueError) as e:
                return self._json({"error": str(e)}, 400)
            self.send_error(HTTPStatus.NOT_FOUND)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"panogen UI → http://127.0.0.1:{port}/   (Ctrl+C 로 종료)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if state["job"]:
            state["job"].cancel()
        httpd.server_close()
