"""이미지 + 프롬프트 입력 준비 (첨부 1~3장).

360° 파노라마(prepare): 첨부 이미지를 프롬프트대로 고쳐 360° 로. 입력 종류에 따라 시작점만 다르다.
- 2:1 파노라마  → 그대로 2048×1024 로 맞춰 img2img (strength = 원본 변경 정도). 추가 이미지는 klein 참조로만
- 일반 사진     → 원근(핀홀) 사진을 equirect 에 투영하고, 사진 밖을 마스크로 표시
                  1장 = 정면, 2장 = 정면·뒤, 3장 = 0°·120°·240° 방향에 나눠 배치
                  → 사진 밖은 새로 생성(outpaint), 사진 영역은 strength 만큼만 프롬프트 반영
일반 이미지(prepare_flat): 첫 장을 목표 비율로 맞춰 잘라 img2img. klein 은 전부 참조 이미지로.
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
    sources: list[np.ndarray] | None = None   # 첨부 원본 그대로 (klein 참조 이미지용, 최대 3장)
    fov: float | None = None  # outpaint: 사진의 가로 화각(도)
    yaws: list[float] | None = None   # outpaint: 사진마다 배치한 방향(도, 0 = 정면)
    unused: int = 0           # 이 모드에서 쓰지 않는 추가 이미지 수 (klein 외 백엔드)

    @property
    def source(self) -> np.ndarray | None:
        return self.sources[0] if self.sources else None


MAX_IMAGES = 3
# 일반 이미지 크기: 약 1MP, 32 배수 (FLUX·Qwen·klein 공통으로 무난한 해상도)
ASPECTS = {"1:1": (1024, 1024), "4:3": (1152, 864), "3:4": (864, 1152), "3:2": (1216, 832),
           "2:3": (832, 1216), "16:9": (1344, 768), "9:16": (768, 1344)}


def flat_size(aspect: str, img: np.ndarray | None = None, base: int = 1024) -> tuple[int, int]:
    """일반 이미지 (가로, 세로). aspect='auto' 면 첨부 첫 장 비율에 가장 가까운 것, 첨부가 없으면 1:1.
    base: 1:1 한 변 (설정 generate.flat_base) — 나머지 비율도 같은 면적으로 맞추고 16 배수로 반올림."""
    if aspect == "auto":
        if img is None:
            aspect = "1:1"
        else:
            r = img.shape[1] / img.shape[0]
            aspect = min(ASPECTS, key=lambda k: abs(np.log(ASPECTS[k][0] / ASPECTS[k][1]) - np.log(r)))
    if aspect not in ASPECTS:
        raise ValueError(f"비율은 auto 또는 {', '.join(ASPECTS)} 중 하나입니다.")
    w, h = ASPECTS[aspect]
    k = base / 1024
    return max(16, round(w * k / 16) * 16), max(16, round(h * k / 16) * 16)


def yaws_for(n: int) -> list[float]:
    """사진 n 장을 고르게 나눠 둘 방향(도). 1장은 정면."""
    return [i * 360.0 / n for i in range(n)]


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
                      erode: int = 4, yaw: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """원근 사진을 equirect 에 투영 (yaw 0 = 가운데 정면, 90 = 오른쪽). (이미지, 마스크[255 = 생성할 곳])."""
    ph, pw = photo.shape[:2]
    lon = (np.arange(out_w) + 0.5) / out_w * 2 * np.pi - np.pi - np.radians(yaw)
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


def _as_list(imgs) -> list[np.ndarray]:
    lst = list(imgs) if isinstance(imgs, (list, tuple)) else [imgs]
    if not 1 <= len(lst) <= MAX_IMAGES:
        raise ValueError(f"첨부 이미지는 1~{MAX_IMAGES}장입니다.")
    return lst


def prepare(imgs, width: int, height: int, strength: float | None = None, fov: float = 90.0) -> InitImage:
    """360° 파노라마용 시작점. imgs: 이미지 1장 또는 최대 3장 목록."""
    from panogen.imageutil import resize

    lst = _as_list(imgs)
    panos = [im for im in lst if is_panorama(im.shape[1], im.shape[0])]
    if panos:   # 파노라마가 있으면 그것을 고쳐 씀 (사진을 섞어 배치할 수는 없음)
        img = panos[0]
        pano = resize(img, width)
        if pano.shape[0] != height:   # 2:1 근처 비율은 정확히 맞춤
            from PIL import Image
            pano = np.asarray(Image.fromarray(pano).resize((width, height), Image.LANCZOS))
        return InitImage("img2img", pano, None, 0.6 if strength is None else strength, lst, unused=len(lst) - 1)
    yaws = yaws_for(len(lst))
    if len(lst) > 1:   # 여러 장이면 서로 겹치지 않게 화각을 제한 (360 / n)
        fov = min(fov, 360.0 / len(lst) - 4)
    erp = np.zeros((height, width, 3), np.float32)
    keep = np.zeros((height, width), bool)
    for im, yaw in zip(lst, yaws):
        h, w = im.shape[:2]
        if max(h, w) > 2048:   # 투영 샘플링용으로 충분한 크기
            im = resize(im, round(w * 2048 / max(h, w)))
        e, m = photo_to_equirect(im, width, height, fov, yaw=yaw)
        inside = m == 0
        erp[inside] = e[inside]
        keep |= inside
    erp[~keep] = np.concatenate([im.reshape(-1, 3) for im in lst]).mean(0)   # 사진 밖은 평균색 (생성 시작점)
    mask = np.where(keep, 0, 255).astype(np.uint8)
    return InitImage("outpaint", np.clip(np.rint(erp), 0, 255).astype(np.uint8), mask,
                     0.3 if strength is None else strength, lst, fov, yaws)


def prepare_flat(imgs, width: int, height: int, strength: float | None = None) -> InitImage:
    """일반 이미지용 시작점: 첫 장을 목표 비율로 가운데 맞춰 잘라 (width, height) 로. 나머지는 klein 참조로만."""
    from PIL import Image

    lst = _as_list(imgs)
    img = lst[0]
    h, w = img.shape[:2]
    r = width / height
    cw, ch = (w, round(w / r)) if w / h < r else (round(h * r), h)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    crop = Image.fromarray(img[y0:y0 + ch, x0:x0 + cw]).resize((width, height), Image.LANCZOS)
    return InitImage("img2img", np.asarray(crop.convert("RGB")), None, 0.6 if strength is None else strength,
                     lst, unused=len(lst) - 1)
