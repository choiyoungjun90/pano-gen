"""파노라마 업스케일.

핵심: 좌우를 순환 패딩(wrap)한 뒤 타일 단위로 처리 → 업스케일 후에도 0°/360° 경계가 이어짐.
- AI 업스케일: spandrel 로 ESRGAN/DAT 등 .pth 모델 로드 (CUDA/MPS/CPU 자동)
- 모델 미지정·torch 미설치 시: Lanczos(비AI) 확대로 대체
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from panogen.imageutil import resize

# 타일 함수: uint8 HxWx3 → uint8 (H*s)x(W*s)x3
TileFn = Callable[[np.ndarray], np.ndarray]


def _ramp(n: int, overlap: int, start_edge: bool, end_edge: bool) -> np.ndarray:
    r = np.ones(n, np.float32)
    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
        if not start_edge:
            r[:overlap] = np.minimum(r[:overlap], ramp)
        if not end_edge:
            r[n - overlap:] = np.minimum(r[n - overlap:], ramp[::-1])
    return r


def _starts(total: int, tile: int, step: int) -> list[int]:
    if total <= tile:
        return [0]
    s = list(range(0, total - tile, step))
    s.append(total - tile)
    return sorted(set(s))


def tiled_upscale(img: np.ndarray, fn: TileFn, scale: int, tile: int = 512,
                  overlap: int = 32, wrap_pad: int = 64,
                  progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    p = min(wrap_pad, w // 2)
    padded = np.concatenate([img[:, w - p:], img, img[:, :p]], axis=1) if p else img
    ph, pw = padded.shape[:2]
    step = max(1, tile - overlap)
    ys, xs = _starts(ph, tile, step), _starts(pw, tile, step)

    out = np.zeros((ph * scale, pw * scale, 3), np.float32)
    wsum = np.zeros((ph * scale, pw * scale, 1), np.float32)
    total, done = len(ys) * len(xs), 0
    for y in ys:
        for x in xs:
            crop = padded[y:y + tile, x:x + tile]
            res = fn(crop).astype(np.float32)
            th, tw = crop.shape[:2]
            assert res.shape[:2] == (th * scale, tw * scale), "업스케일 배율이 설정과 다릅니다"
            ov = overlap * scale
            wy = _ramp(th * scale, ov, y == 0, y + th >= ph)
            wx = _ramp(tw * scale, ov, x == 0, x + tw >= pw)
            wgt = (wy[:, None] * wx[None, :])[..., None]
            ys_, xs_ = y * scale, x * scale
            out[ys_:ys_ + th * scale, xs_:xs_ + tw * scale] += res * wgt
            wsum[ys_:ys_ + th * scale, xs_:xs_ + tw * scale] += wgt
            done += 1
            if progress:
                progress(done, total)
    out /= np.maximum(wsum, 1e-6)
    out = out[:, p * scale:(p + w) * scale]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class TorchUpscaler:
    """spandrel 로 .pth/.safetensors 업스케일 모델을 로드."""

    def __init__(self, model_path: Path, half: bool = True):
        import torch
        from spandrel import ModelLoader

        from panogen.device import pick_device

        self.torch = torch
        self.device = pick_device()
        desc = ModelLoader().load_from_file(str(model_path))
        self.scale = int(desc.scale)
        self.half = half and self.device in ("cuda", "mps") and getattr(desc, "supports_half", False)
        desc.to(self.device).eval()
        if self.half:
            desc.half()
        self.model = desc

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        t = self.torch
        x = t.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)[None].float().div(255)
        x = x.to(self.device)
        if self.half:
            x = x.half()
        with t.inference_mode():
            y = self.model(x)
        y = y.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return np.rint(y * 255).astype(np.uint8)


def upscale(img: np.ndarray, target_width: int, model_path: Path | None = None,
            tile: int = 512, overlap: int = 32, wrap_pad: int = 64, half: bool = True,
            log: Callable[[str], None] = print) -> np.ndarray:
    w = img.shape[1]
    if w >= target_width:
        log(f"이미 {w}px 이상 — 업스케일 생략, 목표 {target_width}px 로 맞춤")
        return resize(img, target_width)

    up = None
    if model_path:
        if not Path(model_path).exists():
            log(f"모델 파일 없음: {model_path} → Lanczos 로 대체")
        else:
            try:
                up = TorchUpscaler(Path(model_path), half=half)
            except ModuleNotFoundError as e:
                log(f"{e.name} 미설치 → Lanczos 로 대체 (requirements 확인)")
    if up is None:
        log(f"Lanczos 확대: {w}px → {target_width}px (비AI, 디테일 생성 없음)")
        return resize(img, target_width)

    log(f"AI 업스케일 x{up.scale} on {up.device} (tile {tile}, wrap {wrap_pad}px)")
    cur = img
    while cur.shape[1] < target_width:
        cur = tiled_upscale(cur, up, up.scale, tile, overlap, wrap_pad,
                            progress=lambda d, t: log(f"  타일 {d}/{t}") if d == t or d % max(1, t // 20) == 0 else None)
    return resize(cur, target_width)
