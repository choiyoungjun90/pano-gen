"""panogen CLI — 로컬 360° 파노라마 생성 (맥/윈도우 공통)

  panogen doctor                           장치·메모리·모델 준비 상태 점검, 프로필 추천
  panogen download                         모델 미리 받기 (FLUX.1-dev, DiT360 LoRA, 업스케일러)
  panogen generate -p "프롬프트"           2048×1024 생성 → 이음새 보정 (2K 까지, --upscale 이면 8K 까지)
  panogen generate --prompts list.txt -c 2 여러 프롬프트 × 시드 2개씩 일괄 생성
  panogen upscale DIR --seed 1 3           2K 확인 후 고른 시드만 8K 로
  panogen upscale IMAGE [IMAGE …]          기존 2K 파노라마를 8K 로
  panogen seam IMAGE [--roll|--unroll F]   이음새 검사·보정
  panogen ui [--port 8370]                 웹 UI (프롬프트 → 진행도 → 2K 확인 → 8K)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from panogen.config import Config, load_config


def _utf8_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def log(msg: str) -> None:
    print(msg, flush=True)


def slug(text: str, n: int = 32) -> str:
    s = re.sub(r"[^\w\-]+", "_", text.strip().lower(), flags=re.UNICODE).strip("_")
    return s[:n] or "pano"


# ---------- 후처리 (이음새 + 8K) ----------

def finish(cfg: Config, img, out_dir: Path, stem: str, upscale_on: bool = True, save_2k: bool = True) -> dict:
    import numpy as np

    from panogen import seam
    from panogen.imageutil import save_rgb
    from panogen.upscale import upscale

    arr = np.asarray(img.convert("RGB")) if hasattr(img, "convert") else img
    s = cfg["seam"]
    rep = seam.check(arr)
    info = {"seam_ratio": round(rep.ratio, 3), "seam_verdict": rep.verdict(float(s["warn_ratio"]))}
    log(f"      이음새 {rep.ratio:.2f}배 → {info['seam_verdict']}")
    if rep.ratio >= float(s["warn_ratio"]):
        arr = seam.blend(arr, int(s["band"]))
        info["seam_after_blend"] = round(seam.check(arr).ratio, 3)
        log(f"      blend 보정 → {info['seam_after_blend']:.2f}배")
    if save_2k:
        save_rgb(arr, out_dir / f"{stem}_2k.png")

    if not upscale_on:
        return info
    u, o = cfg["upscale"], cfg["output"]
    model = cfg.get_path(u["model_path"]) if u.get("model_path") else None
    big = upscale(arr, int(u["target_width"]), model, int(u["tile"]), int(u["overlap"]),
                  int(u["wrap_pad"]), bool(u.get("half", True)), log=lambda m: log("      " + m))
    w = big.shape[1]
    tag = f"{w // 1024}k" if w % 1024 == 0 else f"{w}px"
    jpg = save_rgb(big, out_dir / f"{stem}_{tag}.jpg", int(o["jpeg_quality"]))
    info["output"] = jpg.name
    if o.get("save_png", True):
        save_rgb(big, out_dir / f"{stem}_{tag}.png")
    log(f"      → {jpg}")
    return info


def inpaint_seam_if_needed(cfg: Config, eng, img, emb: dict, seed: int, out_dir: Path | None = None,
                           stem: str = "") -> dict:
    """생성형 이음새 보정 (설정 seam.inpaint). 결과 이미지는 'image' 키로 돌려줌."""
    import numpy as np

    from panogen import seam

    s = cfg["seam"]
    mode = str(s.get("inpaint", "auto"))
    if mode == "off" or not eng.can_inpaint:
        return {"image": img}
    before = seam.check(np.asarray(img.convert("RGB"))).ratio
    if mode == "auto" and before < float(s["warn_ratio"]):
        return {"image": img}
    if out_dir is not None:   # 인페인팅이 실패해도 생성 결과는 남도록 먼저 저장
        img.save(out_dir / f"{stem}_raw.png")
    log(f"      이음새 인페인팅 (비율 {before:.2f}배, strength {s.get('inpaint_strength', 0.5)}) …")
    t0 = time.time()
    img = eng.inpaint_seam(img, emb, seed, float(s.get("inpaint_strength", 0.5)), float(s.get("inpaint_width", 0.1)))
    after = seam.check(np.asarray(img.convert("RGB"))).ratio
    log(f"      이음새 인페인팅 완료 {before:.2f} → {after:.2f}배 ({time.time() - t0:.0f}s)")
    if out_dir is not None:
        (out_dir / f"{stem}_raw.png").unlink(missing_ok=True)
    return {"image": img, "seam_inpaint": {"before": round(before, 3), "after": round(after, 3),
                                           "seconds": round(time.time() - t0, 1)}}


# ---------- 명령 ----------

def cmd_doctor(cfg: Config) -> int:
    import importlib.util
    import os

    from panogen.device import detect
    from panogen.dit360 import pipeline_path
    from panogen.profiles import PROFILES, auto_profile

    d = detect()
    log(f"OS        {d['os']}\nPython    {d['python']}\ntorch     {d.get('torch') or '미설치'}")
    log(f"장치      {d['device']}  {d['gpu'] or ''}")
    if d["vram_gb"]:
        log(f"VRAM      {d['vram_gb']} GB   (bf16 {'지원' if d['bf16'] else '미지원 → fp16'})")
    log(f"RAM       {d['ram_gb']} GB")
    try:
        rec = auto_profile(d["device"], d["vram_gb"])
        chosen = cfg["generate"]["profile"]
        log(f"추천 프로필 {rec.name} — {rec.note}   (현재 설정: {chosen})")
        prof = PROFILES.get(chosen, rec) if chosen != "auto" else rec
        if d["ram_gb"] and d["ram_gb"] < prof.min_ram_gb:
            log(f"  주의: {prof.name} 은 시스템 RAM {prof.min_ram_gb}GB 이상 권장 (현재 {d['ram_gb']}GB)")
        need = {"nf4": "bitsandbytes", "gguf": "gguf"}.get(prof.transformer)
        if need and not importlib.util.find_spec(need):
            log(f"  주의: {prof.name} 에 필요한 `{need}` 패키지가 없습니다")
    except RuntimeError as e:
        log(f"프로필    없음 — {e}")
    ok = pipeline_path(cfg.root).exists()
    log(f"DiT360    {'준비됨' if ok else '없음 → python scripts/fetch_dit360.py'}")
    log(f"HF_TOKEN  {'설정됨' if os.environ.get('HF_TOKEN') else '없음 → .env (FLUX.1-dev 동의 필요)'}")
    up = cfg.get_path(cfg["upscale"]["model_path"])
    log(f"업스케일러 {'준비됨' if up.exists() else '없음 → python scripts/download_upscaler.py (없으면 Lanczos)'}")
    return 0


def cmd_download(cfg: Config, a) -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    from panogen.backends import resolve_backend
    from panogen.device import detect
    from panogen.profiles import resolve

    d = detect()
    prof = resolve(cfg["generate"]["profile"], d["device"], d["vram_gb"])
    b = resolve_backend(cfg, a.backend)
    # 맥(gguf)은 트랜스포머를 GGUF 단일 파일로 받으므로 원본 트랜스포머 가중치는 건너뜀
    ignore = ["*.md", "*.png", "*.jpg"] + (["transformer/*.safetensors", "transformer/*.bin"]
                                           if prof.transformer == "gguf" else [])
    log(f"{b['base']} 받는 중 (백엔드 {b['name']}, 프로필 {prof.name}) — 수십 GB, 시간이 걸립니다")
    snapshot_download(b["base"], ignore_patterns=ignore)
    for key in ("lora", "speed_lora"):
        if not b.get(key) or (key == "speed_lora" and not a.fast):
            continue
        log(f"{key} 받는 중: {b[key]}")
        if b.get(f"{key}_file"):
            hf_hub_download(b[key], b[f"{key}_file"])
        else:
            snapshot_download(b[key])
    if prof.transformer == "gguf":
        log(f"GGUF 받는 중: {b['gguf_file']}")
        hf_hub_download(b["gguf_repo"], b["gguf_file"])
    log("완료. 업스케일러는 python scripts/download_upscaler.py")
    return 0


def cmd_generate(cfg: Config, a) -> int:
    from panogen.device import detect, peak_memory_gb
    from panogen.engine import Engine, Job, build_prompt, make_jobs
    from panogen.profiles import resolve

    g = cfg["generate"]
    for k in ("steps", "seed"):
        if getattr(a, k) is not None:
            g[k] = getattr(a, k)
    prompts = [a.prompt] if a.prompt else [
        ln.strip() for ln in Path(a.prompts).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")]
    if not prompts:
        raise ValueError("프롬프트가 없습니다.")

    d = detect()
    prof = resolve(a.profile or g["profile"], d["device"], d["vram_gb"])
    eng = Engine(cfg.root, cfg, prof, bf16=d["bf16"], log=log, backend=a.backend, fast=a.fast)
    b, s = eng.b, eng.sampling
    log(f"백엔드 {b['name']}{' (fast)' if eng.fast else ''} · 프로필 {prof.name} · {d['gpu']} · "
        f"{int(g['width'])}×{int(g['height'])} · {s['steps']} steps")
    if prof.name == "mps-16g":
        log("  (맥 16GB 는 실험적입니다. 장당 수십 분 걸릴 수 있고, 다른 앱을 닫아 두세요)")

    init, image = None, getattr(a, "image", None)
    if image:
        from panogen import imagein
        from panogen.imageutil import load_rgb

        init = imagein.prepare(load_rgb(Path(image)), int(g["width"]), int(g["height"]), a.strength, a.fov)
        log(f"이미지 입력 {Path(image).name}: "
            + ("2:1 파노라마 → 수정" if init.mode == "img2img" else f"일반 사진 → 360° 확장 (화각 {init.fov:.0f}°)")
            + f", 원본 변경 정도 {init.strength}")
    prefix = (b.get("image_prompt_prefix") or b.get("prompt_prefix", "")) if init else b.get("prompt_prefix", "")
    if getattr(a, "seed_list", None):     # bench: 지정한 시드 목록 그대로
        jobs = [Job(p, s, build_prompt(p, prefix)) for p in prompts for s in a.seed_list]
    else:
        jobs = make_jobs(prompts, int(g["seed"]), a.count, prefix)
    embs = eng.encode(sorted({j.full_prompt for j in jobs}))
    emb_map = dict(zip(sorted({j.full_prompt for j in jobs}), embs))

    upscale_on = a.upscale or (bool(g.get("upscale", False)) and not a.no_upscale)
    out_root = cfg.get_path(cfg["output"]["dir"])
    made: dict[Path, list[int]] = {}
    for i, j in enumerate(jobs, 1):
        name = a.name or f"{time.strftime('%Y%m%d')}_{slug(j.prompt)}"
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        # 한 폴더에 여러 프롬프트가 모이면 파일명이 겹치지 않게 프롬프트 번호를 붙임
        stem = f"seed{j.seed}" if not (a.name and len(prompts) > 1) \
            else f"p{prompts.index(j.prompt) + 1:02d}_seed{j.seed}"
        log(f"[3/3] ({i}/{len(jobs)}) {j.prompt[:60]}  seed={j.seed}")
        t0 = time.time()
        if image and not any(out_dir.glob("input.*")):   # 결과 폴더에 원본 보관 (웹 UI 에서 비교)
            import shutil
            shutil.copy(image, out_dir / f"input{Path(image).suffix.lower()}")
        img = eng.generate(emb_map[j.full_prompt], j.seed, init=init)
        gen_s = time.time() - t0
        log(f"      생성 {gen_s:.0f}s")
        seam_info = inpaint_seam_if_needed(cfg, eng, img, emb_map[j.full_prompt], j.seed, out_dir, stem)
        img = seam_info.pop("image")
        info = {**seam_info, **finish(cfg, img, out_dir, stem, upscale_on=upscale_on)}
        made.setdefault(out_dir, []).append(j.seed)
        meta = {"prompt": j.prompt, "full_prompt": j.full_prompt, "seed": j.seed, "backend": b["name"],
                "fast": eng.fast, "profile": prof.name, **s, "size": [g["width"], g["height"]],
                "model": b["base"], "lora": b["lora"], "generate_seconds": round(gen_s, 1),
                "peak_memory_gb": peak_memory_gb(), "created": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        if init:
            meta["input"] = {"file": Path(image).name, "mode": init.mode, "strength": init.strength, "fov": init.fov}
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log("완료.")
    if not upscale_on:
        log("2K 결과를 확인한 뒤 마음에 드는 시드만 8K 로:")
        for d, seeds in made.items():
            log(f"  panogen upscale {d} --seed {' '.join(str(s) for s in dict.fromkeys(seeds))}")
    return 0


def cmd_bench(cfg: Config, a) -> int:
    """같은 프롬프트·시드로 여러 백엔드를 생성해 outputs/bench_<태그>_<백엔드>/ 에 모음 (2K 만).
    백엔드마다 모델을 한 번만 올리고, 실패한 백엔드는 건너뛰고 계속한다.
    진행 상황은 outputs/bench_<태그>.json 에 기록 (웹 UI 비교 화면이 읽음)."""
    import argparse
    import copy

    from panogen.device import free_memory

    stamp = a.tag or time.strftime("%Y%m%d_%H%M")
    if not re.fullmatch(r"[\w\-]+", stamp):
        raise ValueError("태그는 영문·숫자·_·- 만 쓸 수 있습니다.")
    prompts = [ln.strip() for ln in Path(a.set).read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    if a.limit:
        prompts = prompts[: a.limit]
    if not prompts:
        raise ValueError("프롬프트가 없습니다.")
    out_root = cfg.get_path(cfg["output"]["dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    pfile = out_root / f"bench_{stamp}.prompts.txt"
    pfile.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    runs = []
    for be in [x.strip() for x in a.backends.split(",") if x.strip()]:
        fast = be.endswith(":fast")
        be = be.removesuffix(":fast")
        runs.append({"backend": be, "fast": fast, "dir": f"bench_{stamp}_{be}{'_fast' if fast else ''}",
                     "status": "waiting"})
    summary = {"tag": stamp, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "prompts": prompts,
               "seeds": a.seeds, "runs": runs}
    spath = out_root / f"bench_{stamp}.json"

    def save():
        spath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    for r in runs:
        log(f"===== bench: {r['dir']} =====")
        r["status"] = "running"
        save()
        ga = argparse.Namespace(prompt=None, prompts=pfile, name=r["dir"], count=1, seed=None, steps=None,
                                profile=a.profile, backend=r["backend"], fast=r["fast"], upscale=False,
                                no_upscale=True, seed_list=a.seeds, image=None, strength=None, fov=90.0)
        try:
            cmd_generate(copy.deepcopy(cfg), ga)
            r["status"] = "done"
        except Exception as e:  # 한 백엔드가 실패해도 나머지는 계속
            log(f"오류: {r['backend']} — {type(e).__name__}: {e}")
            r.update(status="failed", error=f"{type(e).__name__}: {e}")
        save()
        free_memory()
    ok = [r["dir"] for r in runs if r["status"] == "done"]
    bad = [r["backend"] for r in runs if r["status"] == "failed"]
    log("벤치 완료: " + ", ".join(ok) + (f"  (실패: {', '.join(bad)})" if bad else ""))
    log("웹 UI(panogen ui) 의 '비교' 탭에서 360° 로 비교하세요.")
    return 1 if bad and not ok else 0


def _pick_2k(targets: list[str], seeds: list[int] | None) -> list[Path]:
    """파일은 그대로, 폴더는 --seed 로 고른 *seedN_2k.png 를 찾는다."""
    out: list[Path] = []
    for t in map(Path, targets):
        if t.is_file():
            out.append(t)
            continue
        if not t.is_dir():
            raise FileNotFoundError(f"없음: {t}")
        found = sorted(t.glob("*seed*_2k.png"))
        avail = sorted({int(m.group(1)) for f in found if (m := re.search(r"seed(\d+)_2k$", f.stem))})
        if not seeds:
            raise ValueError(f"{t} 에서 8K 로 만들 시드를 --seed 로 고르세요. 있는 시드: "
                             f"{' '.join(map(str, avail)) or '(2K 결과 없음)'}")
        for s in seeds:
            hit = [f for f in found if re.search(rf"(^|_)seed{s}_2k$", f.stem)]
            if not hit:
                raise ValueError(f"{t} 에 seed{s} 2K 결과가 없습니다. 있는 시드: {' '.join(map(str, avail))}")
            out += hit
    return out


def _refine_all(cfg: Config, srcs: list[Path], arrs: dict, a) -> dict:
    """--refine: 같은 백엔드끼리 묶어 텍스트 인코딩·모델 적재를 한 번만 하고 2K → 4K 디테일 강화."""
    from panogen.device import detect, free_memory
    from panogen.engine import Engine
    from panogen.profiles import resolve
    from panogen.refine import refine

    r = cfg.get("refine", {})
    strength = float(a.strength if a.strength is not None else r.get("strength", 0.3))
    metas = {}
    for src in srcs:
        mp = src.with_name(f"{src.stem.removesuffix('_2k')}.json")
        m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        if not m.get("prompt"):
            raise ValueError(f"{src.name}: 디테일 강화에는 생성 기록(seedN.json 의 prompt)이 필요합니다.")
        metas[src] = m
    groups: dict[tuple, list[Path]] = {}
    for src, m in metas.items():
        groups.setdefault((m.get("backend", "dit360"), bool(m.get("fast"))), []).append(src)

    d = detect()
    prof = resolve(cfg["generate"]["profile"], d["device"], d["vram_gb"])
    out = {}
    for (backend, fast), group in groups.items():
        eng = Engine(cfg.root, cfg, prof, bf16=d["bf16"], log=log, backend=backend, fast=fast)
        embs = eng.encode([metas[s]["prompt"] for s in group])     # 접두어 없는 원 프롬프트
        for i, (src, emb) in enumerate(zip(group, embs), 1):
            log(f"[refine] ({i}/{len(group)}) {src.name} · {backend}{' (fast)' if fast else ''}")
            t0 = time.time()
            out[src] = (refine(eng, arrs[src], emb, int(metas[src].get("seed", 0)), strength=strength,
                               tile=int(r.get("tile", 1024)), overlap=int(r.get("overlap", 128)),
                               wrap_pad=int(r.get("wrap_pad", 128)), log=lambda m: log("      " + m)),
                        {"refine_strength": strength, "refine_seconds": round(time.time() - t0, 1)})
        del eng
        free_memory()    # ESRGAN 전에 확산 모델 메모리 반환
    return out


def cmd_upscale(cfg: Config, a) -> int:
    from panogen.imageutil import check_ratio, load_rgb

    srcs = _pick_2k(a.targets, a.seed)
    arrs = {}
    for src in srcs:
        arrs[src] = load_rgb(src)
        if (w := check_ratio(arrs[src])):
            log(f"경고: {src.name}: {w}")
    refined = _refine_all(cfg, srcs, arrs, a) if a.refine else {}
    for i, src in enumerate(srcs, 1):
        arr, extra = refined.get(src, (arrs[src], {}))
        out_dir = Path(a.out) if a.out else src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = src.stem.removesuffix("_2k")
        log(f"[upscale] ({i}/{len(srcs)}) {src.name} {arr.shape[1]}×{arr.shape[0]}")
        t0 = time.time()
        info = finish(cfg, arr, out_dir, stem, upscale_on=True, save_2k=False)
        # generate 가 남긴 seedN.json 에 8K 결과를 이어 기록 (이음새 값은 생성 시점 것을 유지)
        meta_p = src.with_name(f"{stem}.json")
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            meta.update(output=info["output"], upscale_seconds=round(time.time() - t0, 1), refined=bool(extra),
                        upscaled=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
            meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def cmd_seam(cfg: Config, a) -> int:
    from panogen import seam
    from panogen.imageutil import load_rgb, save_rgb

    src = Path(a.image)
    band = int(cfg["seam"]["band"])
    if a.unroll:
        img = seam.unroll(load_rgb(src))
        out = save_rgb(img, src.with_name(src.stem.replace("_rolled", "") + "_fixed.png"))
        log(f"되돌림 → {out.name}  (이음새 {seam.check(img).ratio:.2f}배)")
        return 0
    img = load_rgb(src)
    rep = seam.check(img)
    log(f"이음새 {rep.ratio:.2f}배 → {rep.verdict(float(cfg['seam']['warn_ratio']))}")
    if a.roll:
        r = seam.roll(img)
        save_rgb(r, src.with_name(src.stem + "_rolled.png"))
        save_rgb(seam.seam_mask(r, band), src.with_name(src.stem + "_rolled_mask.png"))
        log("가운데 띠를 인페인팅한 뒤:  panogen seam <인페인팅본> --unroll")
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    if sys.platform == "darwin":
        # 맥은 CPU·GPU 가 메모리를 공유하므로 MPS 상한(기본 RAM 의 ~1.1배)을 풀어 스왑으로 넘기게 함.
        # 16GB 에서 Qwen 20B 는 8 step 생성 직후 VAE 디코딩에서 상한에 걸려 실패했음 (48MB 부족)
        import os
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    ap = argparse.ArgumentParser(prog="panogen", description="로컬 360° 파노라마 생성 (DiT360) → 8K")
    ap.add_argument("--config", type=Path, help="추가 설정 파일")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="환경 점검·프로필 추천")
    p = sub.add_parser("download", help="모델 미리 받기 (백엔드별)")
    p.add_argument("--backend", help="dit360 | qwen360 | klein360 (기본: 설정의 model.backend)")
    p.add_argument("--fast", action="store_true", help="속도 LoRA 도 받기")

    p = sub.add_parser("generate", help="파노라마 생성 (2K, --upscale 이면 8K 까지)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("-p", "--prompt")
    src.add_argument("--prompts", type=Path, help="한 줄에 프롬프트 하나 (# 주석 가능)")
    p.add_argument("-n", "--name", help="출력 폴더 이름 (기본: 날짜_프롬프트)")
    p.add_argument("-c", "--count", type=int, default=1, help="프롬프트당 장수 (시드를 바꿔가며)")
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--profile", choices=["auto", "cuda-full", "cuda-24g", "cuda-12g", "mps-16g"])
    p.add_argument("--backend", help="dit360 | qwen360 | klein360 (기본: 설정의 model.backend)")
    p.add_argument("--fast", action="store_true", help="속도 LoRA(예: Qwen Lightning 8 step) 사용 — 빠르지만 품질 확인 필요")
    p.add_argument("--image", type=Path, help="첨부 이미지: 프롬프트대로 고쳐 360° 로 (2:1 파노라마는 수정, 일반 사진은 확장)")
    p.add_argument("--strength", type=float, help="원본 변경 정도 0~1 (기본: 파노라마 0.6, 사진 0.3)")
    p.add_argument("--fov", type=float, default=90.0, help="일반 사진의 가로 화각(도). 폰 기본 카메라 ≈ 70~80, 광각 ≈ 100~120")
    up = p.add_mutually_exclusive_group()
    up.add_argument("--upscale", action="store_true", help="확인 없이 생성 직후 8K 까지")
    up.add_argument("--no-upscale", action="store_true", help="2K 까지만 (설정 generate.upscale = true 를 무시)")

    p = sub.add_parser("bench", help="여러 백엔드를 같은 프롬프트·시드로 비교 생성")
    p.add_argument("--backends", default="dit360,qwen360", help="쉼표 구분, 이름 뒤 :fast 면 속도 LoRA (예: qwen360:fast)")
    p.add_argument("--set", type=Path, default=Path("bench/prompts.txt"), help="프롬프트 파일")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--tag", help="폴더 이름 접두 (기본: 날짜_시각)")
    p.add_argument("--limit", type=int, help="프롬프트 파일의 앞 N개만 (맥에서 빠른 비교용)")
    p.add_argument("--profile", choices=["auto", "cuda-full", "cuda-24g", "cuda-12g", "mps-16g"])

    p = sub.add_parser("upscale", help="2K 파노라마를 8K 로 (폴더 + --seed 로 고르기)")
    p.add_argument("targets", nargs="+", metavar="DIR|IMAGE", help="generate 출력 폴더 또는 2:1 이미지")
    p.add_argument("--seed", type=int, nargs="+", help="폴더에서 8K 로 만들 시드")
    p.add_argument("--refine", action="store_true", help="생성형 디테일 강화 (2K→4K 같은 모델 타일 img2img, 느림)")
    p.add_argument("--strength", type=float, help="디테일 강화 강도 (기본 refine.strength)")
    p.add_argument("-o", "--out")

    p = sub.add_parser("seam", help="이음새 검사·인페인팅 준비")
    p.add_argument("image")
    gr = p.add_mutually_exclusive_group()
    gr.add_argument("--roll", action="store_true")
    gr.add_argument("--unroll", action="store_true")

    p = sub.add_parser("ui", help="웹 UI (프롬프트 → 진행도 → 2K 확인 → 8K)")
    p.add_argument("--port", type=int, default=8370)

    a = ap.parse_args(argv)
    cfg = load_config(extra=a.config)
    try:
        if a.cmd == "doctor":
            return cmd_doctor(cfg)
        if a.cmd == "download":
            return cmd_download(cfg, a)
        if a.cmd == "generate":
            return cmd_generate(cfg, a)
        if a.cmd == "bench":
            return cmd_bench(cfg, a)
        if a.cmd == "upscale":
            return cmd_upscale(cfg, a)
        if a.cmd == "seam":
            return cmd_seam(cfg, a)
        if a.cmd == "ui":
            from panogen.webui import serve

            serve(cfg, a.port, a.config)
            return 0
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        log(f"오류: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
