"""생성형 디테일 강화 (refine) — 2K → 4K 를 같은 모델로 타일 img2img.

Real-ESRGAN 은 있는 픽셀을 선명하게 할 뿐 디테일을 지어내지 않아 8K 에서 뭉개져 보인다.
여기서는 2K 를 Lanczos 로 2배 키운 뒤 1024² 타일마다 낮은 strength(0.25~0.35)로 다시 그려
질감·잔디테일을 만든다. 이후 기존 ESRGAN 으로 8K.

- 좌우 순환 패딩·타일 블렌딩은 upscale.tiled_upscale 재사용 → 0°/360° 경계 유지
- 극점(위·아래)은 equirect 에서 크게 늘어나 있어 타일로 다시 그리면 360° 에서 꼬임이 생김
  → 위도 ±lat_full° 까지만 리파인하고 ±lat_zero° 까지 원본으로 서서히 섞음
- 파노라마 LoRA 는 끔: 타일은 일반 원근 이미지 조각이라 360 LoRA 가 오히려 왜곡을 넣을 수 있음
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from panogen.imageutil import resize
from panogen.upscale import tiled_upscale

from panogen.engine import IMG2IMG  # noqa: E402


def lat_weight(h: int, lat_full: float = 60.0, lat_zero: float = 75.0) -> np.ndarray:
    """행별 리파인 가중치 (1 = 리파인본, 0 = 원본)."""
    lat = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    return np.clip((lat_zero - np.abs(lat)) / max(1e-6, lat_zero - lat_full), 0, 1).astype(np.float32)


def refine_array(img: np.ndarray, tile_fn: Callable[[np.ndarray], np.ndarray], tile: int = 1024,
                 overlap: int = 128, wrap_pad: int = 128, lat_full: float = 60.0, lat_zero: float = 75.0,
                 progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """이미 목표 크기로 키운 img 의 중위도 띠만 tile_fn 으로 다시 그려 합성."""
    h = img.shape[0]
    wgt = lat_weight(h, lat_full, lat_zero)
    rows = np.nonzero(wgt > 0)[0]
    y0 = int(rows[0]) // 16 * 16
    y1 = min(h, -(-(int(rows[-1]) + 1) // 16) * 16)
    band = img[y0:y1]
    tile = min(tile, band.shape[0] // 16 * 16)    # 띠가 타일보다 낮으면 타일을 줄임 (16 배수 유지)
    out_band = tiled_upscale(band, tile_fn, 1, tile=tile, overlap=min(overlap, tile // 4),
                             wrap_pad=wrap_pad, progress=progress)
    res = img.astype(np.float32)
    w = wgt[y0:y1, None, None]
    res[y0:y1] = out_band.astype(np.float32) * w + res[y0:y1] * (1 - w)
    return np.clip(np.rint(res), 0, 255).astype(np.uint8)


def refine(eng, img_2k: np.ndarray, emb: dict, seed: int, scale: int = 2, strength: float = 0.3,
           tile: int = 1024, overlap: int = 128, wrap_pad: int = 128, log=print) -> np.ndarray:
    """eng: 트랜스포머가 올라간 Engine. emb: 접두어 없는 원 프롬프트 임베딩."""
    from PIL import Image

    kind = eng.b["kind"]
    if kind not in IMG2IMG:
        raise RuntimeError(f"백엔드 {eng.b['name']} 은 디테일 강화를 지원하지 않습니다 (img2img 파이프라인 없음).")
    t = eng.torch
    pipe = eng.load()
    big = resize(img_2k, img_2k.shape[1] * scale)
    active = list(pipe.get_active_adapters()) if hasattr(pipe, "get_active_adapters") else []
    if "pano" in active:
        pipe.set_adapters(active, adapter_weights=[0.0 if a == "pano" else 1.0 for a in active])
    ip = eng.sibling(IMG2IMG[kind])
    ip.set_progress_bar_config(disable=True)
    dev = eng.profile.device
    cond = {k: v.to(dev) for k, v in emb.items()}
    n = {"i": 0}

    def tile_fn(crop: np.ndarray) -> np.ndarray:
        n["i"] += 1
        with t.inference_mode():
            out = ip(**cond, image=Image.fromarray(crop), strength=strength, width=crop.shape[1],
                     height=crop.shape[0], num_inference_steps=eng.sampling["steps"], **eng._call_kwargs(),
                     generator=t.Generator("cpu").manual_seed(seed + n["i"])).images[0]
        return np.asarray(out.convert("RGB"))

    log(f"디테일 강화: {img_2k.shape[1]}px → {big.shape[1]}px, 타일 {tile}, strength {strength}, "
        f"{eng.sampling['steps']} steps × {strength}")
    try:
        return refine_array(big, tile_fn, tile, overlap, wrap_pad,
                            progress=lambda d, tot: log(f"  리파인 타일 {d}/{tot}"))
    finally:
        if "pano" in active:
            pipe.set_adapters(active, adapter_weights=[float(eng.b["lora_scale"]) if a == "pano" else 1.0
                                                       for a in active])
