"""좌우 이음새(0°/360°) 검사·보정.

- check:  경계 열 차이를 내부 인접 열 차이와 비교해 점수화
- blend:  경계 양쪽에 색 차이를 반씩 나눠 서서히 흡수 (톤 단차용, 비AI)
- roll:   가로 50% 이동 → 경계가 가운데로 옴. 외부 인페인팅용 마스크도 생성
          구조가 어긋난 이음새는 roll 후 ComfyUI 등에서 인페인팅 → unroll
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeamReport:
    edge_diff: float      # 왼쪽 끝 열 vs 오른쪽 끝 열 평균 절대차
    interior_diff: float  # 내부 인접 열 평균 절대차
    ratio: float

    def verdict(self, warn_ratio: float) -> str:
        if self.ratio < warn_ratio:
            return "양호"
        if self.ratio < warn_ratio * 2:
            return "주의 — blend 로 보정 권장"
        return "불량 — roll 후 인페인팅 권장"


def check(img: np.ndarray) -> SeamReport:
    a = img.astype(np.float32)
    edge = float(np.abs(a[:, 0] - a[:, -1]).mean())
    interior = float(np.abs(np.diff(a, axis=1)).mean())
    return SeamReport(edge, interior, edge / max(interior, 1e-6))


def blend(img: np.ndarray, band: int = 48) -> np.ndarray:
    """경계 색 단차를 양쪽 band 폭에 걸쳐 선형으로 분산시켜 제거."""
    h, w = img.shape[:2]
    band = max(1, min(band, w // 4))
    a = img.astype(np.float32)
    # 행별 경계 차이 (세로로 약간 평활화해 노이즈 억제)
    d = a[:, 0] - a[:, -1]
    k = 9
    kernel = np.ones(k, np.float32) / k
    d = np.stack([np.convolve(np.pad(d[:, c], k // 2, mode="edge"), kernel, mode="valid")
                  for c in range(d.shape[1])], axis=1)
    ramp = (1.0 - np.arange(band, dtype=np.float32) / band)[None, :, None]  # 1 → 0
    half = d[:, None, :] * 0.5
    a[:, :band] -= half * ramp                      # 왼쪽 끝을 오른쪽 색 쪽으로
    a[:, w - band:] += half * ramp[:, ::-1]         # 오른쪽 끝을 왼쪽 색 쪽으로
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def roll(img: np.ndarray) -> np.ndarray:
    return np.roll(img, img.shape[1] // 2, axis=1)


def unroll(img: np.ndarray) -> np.ndarray:
    return np.roll(img, -(img.shape[1] // 2), axis=1)


def seam_mask(img: np.ndarray, band: int = 48) -> np.ndarray:
    """roll 된 이미지 기준, 가운데 세로 띠가 흰색인 인페인팅 마스크."""
    h, w = img.shape[:2]
    m = np.zeros((h, w, 3), np.uint8)
    c = w // 2
    m[:, max(0, c - band): c + band] = 255
    return m
