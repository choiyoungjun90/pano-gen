"""panogen CLI — 로컬 360° 파노라마 생성 (맥/윈도우 공통)

  panogen doctor                           장치·메모리·모델 준비 상태 점검, 프로필 추천
  panogen download                         모델 미리 받기 (FLUX.1-dev, DiT360 LoRA, 업스케일러)
  panogen generate -p "프롬프트"           2048×1024 생성 → 이음새 보정 (2K 까지, --upscale 이면 8K 까지)
  panogen generate --prompts list.txt -c 2 여러 프롬프트 × 시드 2개씩 일괄 생성
  panogen generate -p … --image a.jpg --image b.jpg   첨부 최대 3장 (프롬프트대로 고쳐서)
  panogen generate -p … --type flat --aspect 16:9     일반 이미지 (360 아님)
  panogen generate -p … --seam-later       이음새 보정 전 초안(seedN_raw.png)까지만 → 확인 후 upscale
  panogen upscale DIR --seed 1 3           2K 확인 후 고른 시드만 8K 로 (초안이면 이음새 보정부터)
  panogen upscale DIR --seed 1 --seam-only 초안의 이음새만 보정 (seedN_2k.png)
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

def seam_info(cfg: Config, arr) -> dict:
    """이음새 검사만 (보정 없음) — 초안에도 판정을 붙여 확인 화면에 보여 줌."""
    from panogen import seam

    rep = seam.check(arr)
    return {"seam_ratio": round(rep.ratio, 3), "seam_verdict": rep.verdict(float(cfg["seam"]["warn_ratio"]))}


def finish(cfg: Config, img, out_dir: Path, stem: str, upscale_on: bool = True, save_2k: bool = True,
           pano: bool = True) -> dict:
    """2K 저장(파노라마면 이음새 검사·blend) → 선택: 업스케일. 일반 이미지는 이음새 없이 flat_scale 배로."""
    import numpy as np

    from panogen import seam
    from panogen.imageutil import save_rgb
    from panogen.upscale import upscale

    arr = np.asarray(img.convert("RGB")) if hasattr(img, "convert") else img
    s = cfg["seam"]
    info = {}
    if pano:
        info = seam_info(cfg, arr)
        log(f"      이음새 {info['seam_ratio']:.2f}배 → {info['seam_verdict']}")
        if info["seam_ratio"] >= float(s["warn_ratio"]):
            arr = seam.blend(arr, int(s["band"]))
            info["seam_after_blend"] = round(seam.check(arr).ratio, 3)
            log(f"      blend 보정 → {info['seam_after_blend']:.2f}배")
    if save_2k:
        save_rgb(arr, out_dir / f"{stem}_2k.png")

    if not upscale_on:
        return info
    u, o = cfg["upscale"], cfg["output"]
    model = cfg.get_path(u["model_path"]) if u.get("model_path") else None
    target = int(u["target_width"]) if pano else arr.shape[1] * int(u.get("flat_scale", 4))
    big = upscale(arr, target, model, int(u["tile"]), int(u["overlap"]),
                  int(u["wrap_pad"]) if pano else 0, bool(u.get("half", True)), log=lambda m: log("      " + m))
    w = big.shape[1]
    tag = f"{w // 1024}k" if w % 1024 == 0 else f"{w}px"
    jpg = save_rgb(big, out_dir / f"{stem}_{tag}.jpg", int(o["jpeg_quality"]))
    info["output"] = jpg.name
    if o.get("save_png", True):
        save_rgb(big, out_dir / f"{stem}_{tag}.png")
    log(f"      → {jpg}")
    return info


def inpaint_seam_if_needed(cfg: Config, eng, img, emb: dict, seed: int, out_dir: Path | None = None,
                           stem: str = "", mode: str | None = None) -> dict:
    """생성형 이음새 보정 (설정 seam.inpaint, mode 로 덮어쓰기). 결과 이미지는 'image' 키로 돌려줌."""
    import numpy as np

    from panogen import seam

    s = cfg["seam"]
    mode = mode or str(s.get("inpaint", "auto"))
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
    pano = getattr(a, "type", "pano") != "flat"
    images = [Path(x) for x in (getattr(a, "image", None) or [])]
    if len(images) > 3:
        raise ValueError("첨부 이미지는 최대 3장입니다.")

    from panogen import imagein
    from panogen.imageutil import load_rgb

    arrs = [load_rgb(p) for p in images]
    size = (int(g["width"]), int(g["height"])) if pano else imagein.flat_size(
        getattr(a, "aspect", "auto") or "auto", arrs[0] if arrs else None, int(g.get("flat_base", 1024)))

    d = detect()
    prof = resolve(a.profile or g["profile"], d["device"], d["vram_gb"])
    eng = Engine(cfg.root, cfg, prof, bf16=d["bf16"], log=log, backend=a.backend, fast=a.fast, pano=pano)
    b, s = eng.b, eng.sampling
    log(f"백엔드 {b['name']}{' (fast)' if eng.fast else ''} · 프로필 {prof.name} · {d['gpu']} · "
        f"{'360° 파노라마' if pano else '일반 이미지'} {size[0]}×{size[1]} · {s['steps']} steps")
    if prof.name == "mps-16g":
        log("  (맥 16GB 는 실험적입니다. 장당 수십 분 걸릴 수 있고, 다른 앱을 닫아 두세요)")

    init = None
    if arrs:
        if pano:
            init = imagein.prepare(arrs, size[0], size[1], a.strength, a.fov)
            kind = ("2:1 파노라마 → 수정" if init.mode == "img2img"
                    else f"사진 {len(init.yaws)}장 → 360° 확장 (화각 {init.fov:.0f}°)")
        else:
            init = imagein.prepare_flat(arrs, size[0], size[1], a.strength)
            kind = "첫 장을 바탕으로 수정"
        if b["kind"] == "klein":
            kind = f"참조 이미지 {len(arrs)}장"
        log(f"이미지 입력 {', '.join(p.name for p in images)}: {kind}, 원본 변경 정도 {init.strength}")
    if pano:
        prefix = (b.get("image_prompt_prefix") or b.get("prompt_prefix", "")) if init else b.get("prompt_prefix", "")
    else:
        prefix = ""                               # 파노라마 트리거 문구 없이 프롬프트 그대로
    if getattr(a, "seed_list", None):     # bench: 지정한 시드 목록 그대로
        jobs = [Job(p, s, build_prompt(p, prefix)) for p in prompts for s in a.seed_list]
    else:
        jobs = make_jobs(prompts, int(g["seed"]), a.count, prefix)
    embs = eng.encode(sorted({j.full_prompt for j in jobs}))
    emb_map = dict(zip(sorted({j.full_prompt for j in jobs}), embs))

    upscale_on = a.upscale or (bool(g.get("upscale", False)) and not a.no_upscale)
    draft = pano and getattr(a, "seam_later", False)    # 이음새 보정 전 초안까지만 (확인 후 upscale)
    if draft:
        upscale_on = False
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
        if images and not any(out_dir.glob("input*.*")):   # 결과 폴더에 원본 보관 (웹 UI 에서 비교)
            import shutil
            for k, p in enumerate(images, 1):
                shutil.copy(p, out_dir / f"input{k}{p.suffix.lower()}")
        img = eng.generate(emb_map[j.full_prompt], j.seed, init=init, size=size)
        gen_s = time.time() - t0
        log(f"      생성 {gen_s:.0f}s")
        if draft:
            import numpy as np

            img.save(out_dir / f"{stem}_raw.png")
            info = {**seam_info(cfg, np.asarray(img.convert("RGB"))), "seam_pending": True}
            log(f"      초안 저장 (이음새 {info['seam_ratio']:.2f}배 → {info['seam_verdict']}) — 확인 후 이음새 보정")
        elif pano:
            seam_i = inpaint_seam_if_needed(cfg, eng, img, emb_map[j.full_prompt], j.seed, out_dir, stem)
            img = seam_i.pop("image")
            info = {**seam_i, **finish(cfg, img, out_dir, stem, upscale_on=upscale_on)}
        else:
            info = finish(cfg, img, out_dir, stem, upscale_on=upscale_on, pano=False)
        made.setdefault(out_dir, []).append(j.seed)
        meta = {"prompt": j.prompt, "full_prompt": j.full_prompt, "seed": j.seed, "backend": b["name"],
                "type": "pano" if pano else "flat", "fast": eng.fast, "profile": prof.name, **s,
                "size": list(size), "model": b["base"], "lora": b["lora"] if pano else "",
                "generate_seconds": round(gen_s, 1), "peak_memory_gb": peak_memory_gb(),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        if init:
            meta["input"] = {"files": [p.name for p in images], "file": images[0].name, "mode": init.mode,
                             "strength": init.strength, "fov": init.fov, "yaws": init.yaws, "unused": init.unused}
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log("완료.")
    if not upscale_on:
        log("결과를 확인한 뒤 마음에 드는 시드만 " + ("이음새 보정·8K 로:" if draft else "고해상도로:"))
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
                                no_upscale=True, seed_list=a.seeds, image=None, strength=None, fov=90.0,
                                type="pano", aspect="auto", seam_later=False)
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
    """파일은 그대로, 폴더는 --seed 로 고른 *seedN_2k.png 를 찾는다 (없으면 이음새 보정 전 초안 *seedN_raw.png)."""
    out: list[Path] = []
    for t in map(Path, targets):
        if t.is_file():
            out.append(t)
            continue
        if not t.is_dir():
            raise FileNotFoundError(f"없음: {t}")
        found = sorted(t.glob("*seed*_2k.png")) + sorted(t.glob("*seed*_raw.png"))
        avail = sorted({int(m.group(1)) for f in found if (m := re.search(r"seed(\d+)_(2k|raw)$", f.stem))})
        if not seeds:
            raise ValueError(f"{t} 에서 8K 로 만들 시드를 --seed 로 고르세요. 있는 시드: "
                             f"{' '.join(map(str, avail)) or '(2K 결과 없음)'}")
        for s in seeds:
            hit = [f for f in found if re.search(rf"(^|_)seed{s}_2k$", f.stem)]
            have = {f.stem.removesuffix("_2k") for f in hit}
            hit += [f for f in found if re.search(rf"(^|_)seed{s}_raw$", f.stem) and f.stem.removesuffix("_raw") not in have]
            if not hit:
                raise ValueError(f"{t} 에 seed{s} 2K 결과가 없습니다. 있는 시드: {' '.join(map(str, avail))}")
            out += hit
    return out


def _stem(src: Path) -> str:
    return src.stem.removesuffix("_2k").removesuffix("_raw")


def _meta(src: Path) -> dict:
    mp = src.with_name(f"{_stem(src)}.json")
    return json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}


def _seam_fix_all(cfg: Config, srcs: list[Path], arrs: dict, force: bool) -> dict[Path, Path]:
    """이음새 보정 전 초안(*_raw.png)을 보정해 *_2k.png 로. 같은 백엔드끼리 모델을 한 번만 올림.
    force: 비율이 낮아도 생성형 인페인팅 (사용자가 직접 고른 경우). 반환: 초안 → 보정본 경로."""
    from panogen.device import detect, free_memory
    from panogen.engine import Engine
    from panogen.profiles import resolve

    groups: dict[tuple, list[Path]] = {}
    for src in srcs:
        m = _meta(src)
        groups.setdefault((m.get("backend", "dit360"), bool(m.get("fast"))), []).append(src)
    d = detect()
    prof = resolve(cfg["generate"]["profile"], d["device"], d["vram_gb"])
    out = {}
    for (backend, fast), group in groups.items():
        eng = Engine(cfg.root, cfg, prof, bf16=d["bf16"], log=log, backend=backend, fast=fast)
        metas = [_meta(s) for s in group]
        prompts = [m.get("full_prompt") or m.get("prompt") or "" for m in metas]
        embs = eng.encode(prompts)
        for i, (src, m, emb) in enumerate(zip(group, metas, embs), 1):
            log(f"[seam] ({i}/{len(group)}) {src.name} · {backend}{' (fast)' if fast else ''}")
            from PIL import Image

            # DiT360 텍스트 생성본은 추론 중 순환 패딩으로 이미 이어져 있어 인페인팅 대신 blend 만
            eng.circular = eng.b["kind"] == "dit360" and not m.get("input")
            img = Image.fromarray(arrs[src])
            si = inpaint_seam_if_needed(cfg, eng, img, emb, int(m.get("seed", 0)),
                                        mode="always" if force else None)
            stem = _stem(src)
            info = {**{k: v for k, v in si.items() if k != "image"},
                    **finish(cfg, si["image"], src.parent, stem, upscale_on=False)}
            dst = src.with_name(f"{stem}_2k.png")
            from panogen.imageutil import load_rgb

            arrs[dst] = load_rgb(dst)
            mp = src.with_name(f"{stem}.json")
            if mp.exists():
                meta = json.loads(mp.read_text(encoding="utf-8"))
                meta.update(seam_pending=False, seam_fixed=time.strftime("%Y-%m-%d %H:%M:%S"), **info)
                mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            out[src] = dst
        del eng
        free_memory()
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
        m = _meta(src)
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
        if _meta(src).get("type", "pano") == "pano" and (w := check_ratio(arrs[src])):
            log(f"경고: {src.name}: {w}")
    drafts = [s for s in srcs if s.stem.endswith("_raw")]
    if drafts:      # 이음새 보정 전 초안 → 먼저 보정 (확인 화면에서 직접 고른 경우는 생성형 인페인팅 강제)
        fixed = _seam_fix_all(cfg, drafts, arrs, force=getattr(a, "seam_only", False))
        srcs = [fixed.get(s, s) for s in srcs]
    if getattr(a, "seam_only", False):
        log("이음새 보정 완료.")
        return 0
    flat = [s for s in srcs if _meta(s).get("type") == "flat"]
    if a.refine and flat:
        log(f"디테일 강화는 360° 파노라마만 지원 — 일반 이미지 {len(flat)}장은 업스케일만 합니다")
    refined = _refine_all(cfg, [s for s in srcs if s not in flat], arrs, a) if a.refine else {}
    for i, src in enumerate(srcs, 1):
        arr, extra = refined.get(src, (arrs[src], {}))
        out_dir = Path(a.out) if a.out else src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _stem(src)
        log(f"[upscale] ({i}/{len(srcs)}) {src.name} {arr.shape[1]}×{arr.shape[0]}")
        t0 = time.time()
        info = finish(cfg, arr, out_dir, stem, upscale_on=True, save_2k=False, pano=src not in flat)
        # generate 가 남긴 seedN.json 에 결과를 이어 기록 (이음새 값은 생성·보정 시점 것을 유지)
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
    p.add_argument("--type", choices=["pano", "flat"], default="pano", help="pano = 360° 파노라마(기본), flat = 일반 이미지")
    p.add_argument("--aspect", default="auto", help="일반 이미지 비율: auto(첨부 첫 장 비율, 없으면 1:1) | 1:1 4:3 3:4 3:2 2:3 16:9 9:16")
    p.add_argument("--image", type=Path, action="append", help="첨부 이미지 (최대 3번): 프롬프트대로 고쳐서. "
                   "360 이면 2:1 파노라마는 수정, 일반 사진은 방향을 나눠 배치 후 확장 / klein 은 전부 참조")
    p.add_argument("--strength", type=float, help="원본 변경 정도 0~1 (기본: 파노라마·일반 이미지 0.6, 사진→360 0.3)")
    p.add_argument("--fov", type=float, default=90.0, help="일반 사진의 가로 화각(도). 폰 기본 카메라 ≈ 70~80, 광각 ≈ 100~120")
    up = p.add_mutually_exclusive_group()
    up.add_argument("--upscale", action="store_true", help="확인 없이 생성 직후 8K 까지")
    up.add_argument("--no-upscale", action="store_true", help="2K 까지만 (설정 generate.upscale = true 를 무시)")
    p.add_argument("--seam-later", action="store_true", help="360: 이음새 보정 전 초안(seedN_raw.png)까지만 — 확인 후 upscale")

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
    p.add_argument("--seam-only", action="store_true", help="초안(*_raw.png)의 이음새만 보정하고 끝 (8K 안 함)")
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
