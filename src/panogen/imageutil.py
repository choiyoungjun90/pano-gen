"""이미지 입출력 공통 함수 (numpy uint8 HxWx3 기준)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# 8K 이상 파노라마는 Pillow 기본 '압축 폭탄' 경고 한도를 넘으므로 해제
Image.MAX_IMAGE_PIXELS = None


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def save_rgb(arr: np.ndarray, path: str | Path, quality: int = 95) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    if path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(path, quality=quality, subsampling=0, optimize=True)
    else:
        im.save(path)
    return path


def resize(arr: np.ndarray, width: int) -> np.ndarray:
    """2:1 비율 유지하며 Lanczos 리사이즈."""
    h, w = arr.shape[:2]
    if w == width:
        return arr
    height = round(h * width / w)
    im = Image.fromarray(arr).resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def check_ratio(arr: np.ndarray) -> str | None:
    h, w = arr.shape[:2]
    if abs(w / h - 2.0) > 0.01:
        return f"2:1 비율이 아닙니다 ({w}×{h}). 등장방형 파노라마가 맞는지 확인하세요."
    return None
