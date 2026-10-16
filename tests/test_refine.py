import numpy as np

from panogen.refine import lat_weight, refine_array


def test_lat_weight_keeps_poles():
    w = lat_weight(180, 60, 75)            # 1 행 = 1°
    assert w[90] == 1 and w[30] == 1        # 적도·위도 60° 근처는 리파인
    assert w[0] == 0 and w[-1] == 0         # 극점은 원본
    assert 0 < w[20] < 1                    # 60~75° 사이는 섞임


def test_refine_array_band_and_wrap():
    rng = np.random.default_rng(0)
    h, w = 256, 512
    img = rng.integers(0, 200, (h, w, 3), dtype=np.uint8)
    out = refine_array(img, lambda c: np.clip(c.astype(int) + 40, 0, 255).astype(np.uint8),
                       tile=128, overlap=32, wrap_pad=32)
    assert out.shape == img.shape
    assert np.array_equal(out[:5], img[:5]) and np.array_equal(out[-5:], img[-5:])     # 극점 그대로
    mid = slice(h // 2 - 10, h // 2 + 10)
    assert np.allclose(out[mid].astype(int) - img[mid].astype(int), 40, atol=1)        # 중위도는 다시 그림
    # 좌우 경계: 원본과 같은 연속성 (리파인이 경계에 단차를 만들지 않음)
    d_edge = np.abs(out[mid, 0].astype(int) - out[mid, -1].astype(int)).mean()
    d_src = np.abs(img[mid, 0].astype(int) - img[mid, -1].astype(int)).mean()
    assert abs(d_edge - d_src) < 2
