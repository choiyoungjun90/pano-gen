import numpy as np

from panogen import seam
from panogen.upscale import tiled_upscale


def nearest2x(a):
    return a.repeat(2, axis=0).repeat(2, axis=1)


def test_tiled_matches_full_image():
    img = np.random.default_rng(1).integers(0, 255, (96, 192, 3), dtype=np.uint8)
    out = tiled_upscale(img, nearest2x, 2, tile=64, overlap=16, wrap_pad=16)
    assert out.shape == (192, 384, 3)
    assert np.array_equal(out, nearest2x(img))


def blur2x(a):
    """주변 픽셀을 섞는 업스케일러 흉내 — 경계 처리가 이음새에 영향을 줌."""
    up = nearest2x(a).astype(np.float32)
    k = np.ones(5, np.float32) / 5
    for ax in (0, 1):
        up = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 2, mode="edge"), k, "valid"), ax, up)
    return np.clip(np.rint(up), 0, 255).astype(np.uint8)


def test_wrap_padding_keeps_seam_continuous():
    x = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    row = (127 + 120 * np.sign(np.sin(3 * x)))[None, :, None]
    img = np.repeat(np.repeat(row, 64, 0), 3, 2).astype(np.uint8)
    wrapped = tiled_upscale(img, blur2x, 2, tile=64, overlap=8, wrap_pad=16)
    unwrapped = tiled_upscale(img, blur2x, 2, tile=64, overlap=8, wrap_pad=0)
    assert seam.check(wrapped).ratio <= seam.check(unwrapped).ratio + 1e-6
