"""이미지 + 프롬프트 입력 준비.

첨부 이미지를 프롬프트대로 고쳐 360° 파노라마로 만든다. 입력 종류에 따라 시작점만 다르다.
- 2:1 파노라마  → 그대로 2048×1024 로 맞춰 img2img (strength = 원본 변경 정도)
- 일반 사진     → 원근(핀홀) 사진을 equirect 정면에 투영하고, 사진 밖을 마스크로 표시
                  → 사진 밖은 새로 생성(outpaint), 사진 영역은 strength 만큼만 프롬프트 반영
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InitImage:
    mode: str                 # "img2img" | "outpaint"
    image: np.ndarray         # HxWx3 uint8, 생성 해상도(equirect)
    mask: np.ndarray | None   # HxW uint8, 255 = 새로 그릴 곳 (outpaint 만)
    strength: float           # 원본 변경 정도 0~1
    source: np.ndarray | None = None   # 원본 그대로 (klein 참조 이미지용)
    fov: float | None = None  # outpaint: 사진의 가로 화각(도)


def is_panorama(w: int, h: int) -> bool:
    return 1.9 <= w / h <= 2.1


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[..., None], (v - y0)[..., None]
    a = img.astype(np.float32)
    top = a[y0, x0] * (1 - fx) + a[y0, x0 + 1] * fx
    bot = a[y0 + 1, x0] * (1 - fx) + a[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def photo_to_equirect(photo: np.ndarray, out_w: int, out_h: int, hfov: float = 90.0,
                      erode: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """원근 사진을 equirect 정면(경도 0 = 가운데)에 투영. (이미지, 마스크[255 = 생성할 곳])."""
    ph, pw = photo.shape[:2]
    lon = (np.arange(out_w) + 0.5) / out_w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(out_h) + 0.5) / out_h * np.pi
    lon, lat = np.meshgrid(lon, lat)
    dx, dy, dz = np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    f = (pw / 2) / np.tan(np.radians(hfov) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = pw / 2 + f * dx / dz - 0.5
        v = ph / 2 - f * dy / dz - 0.5
    inside = (dz > 1e-6) & (u >= 0) & (u <= pw - 1) & (v >= 0) & (v <= ph - 1)
    if erode > 0:   # 사진 가장자리 몇 픽셀은 새로 그려 경계를 자연스럽게
        k = inside.copy()
        for _ in range(erode):
            k[1:-1, 1:-1] &= k[:-2, 1:-1] & k[2:, 1:-1] & k[1:-1, :-2] & k[1:-1, 2:]
        inside = k
    out = np.empty((out_h, out_w, 3), np.float32)
    out[:] = photo.reshape(-1, 3).mean(0)          # 사진 밖은 평균색으로 채워 둠 (생성 시작점)
    out[inside] = _bilinear(photo, u[inside], v[inside])
    mask = np.where(inside, 0, 255).astype(np.uint8)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), mask


def prepare(img: np.ndarray, width: int, height: int, strength: float | None = None,
            fov: float = 90.0) -> InitImage:
    from panogen.imageutil import resize

    h, w = img.shape[:2]
    if is_panorama(w, h):
        pano = resize(img, width)
        if pano.shape[0] != height:   # 2:1 근처 비율은 정확히 맞춤
            from PIL import Image
            pano = np.asarray(Image.fromarray(pano).resize((width, height), Image.LANCZOS))
        return InitImage("img2img", pano, None, 0.6 if strength is None else strength, img)
    src = img
    if max(h, w) > 2048:   # 투영 샘플링용으로 충분한 크기
        img = resize(img, round(w * 2048 / max(h, w)))
    erp, mask = photo_to_equirect(img, width, height, fov)
    return InitImage("outpaint", erp, mask, 0.3 if strength is None else strength, src, fov)
