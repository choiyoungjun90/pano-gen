import numpy as np
import pytest

from panogen import seam


def smooth_pano(w=256, h=128):
    x = np.linspace(0, 2 * np.pi, w, endpoint=False)
    row = (127 + 100 * np.sin(x))[None, :, None]
    return np.repeat(np.repeat(row, h, 0), 3, 2).astype(np.uint8)


def test_seamless_image_scores_low():
    rep = seam.check(smooth_pano())
    assert rep.ratio < 2.0


def test_blend_removes_tone_step():
    img = smooth_pano().astype(np.int16)
    img[:, -40:] += 40                     # 오른쪽 끝 톤 단차
    img = np.clip(img, 0, 255).astype(np.uint8)
    before = seam.check(img).ratio
    after = seam.check(seam.blend(img, band=48)).ratio
    assert before > 5 and after < 2.0


def test_roll_unroll_roundtrip():
    img = np.random.default_rng(0).integers(0, 255, (64, 128, 3), dtype=np.uint8)
    assert np.array_equal(seam.unroll(seam.roll(img)), img)


def test_inpaint_seam_only_touches_wrap_band(monkeypatch):
    """경계를 가운데로 옮겨 띠만 다시 그리고 되돌림 → 좌우 끝만 바뀌고 나머지는 원본 그대로."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from PIL import Image

    from panogen.engine import Engine
    from panogen.profiles import Profile

    calls = {}

    class FakeInpaint:
        def __init__(self, **components):
            pass

        def set_progress_bar_config(self, **kw):
            pass

        def __call__(self, image, mask_image, strength, **kw):
            calls.update(strength=strength, steps=kw["num_inference_steps"], mask=np.asarray(mask_image))
            return type("R", (), {"images": [Image.new("RGB", image.size, (255, 0, 0))]})()

    monkeypatch.setattr(diffusers, "QwenImageInpaintPipeline", FakeInpaint, raising=False)
    eng = Engine.__new__(Engine)
    eng.torch, eng.b = torch, {"kind": "qwen"}
    eng.pipe = type("P", (), {"components": {"transformer": None, "vae": None}})()
    eng.profile = Profile("t", "cpu", "bf16", "cpu", False, 0, "")
    eng.sampling = {"steps": 8, "guidance": 1.0, "true_cfg": 1.0}

    w, h = 200, 100
    src = np.full((h, w, 3), 40, np.uint8)
    out = np.asarray(eng.inpaint_seam(Image.fromarray(src), {}, seed=1, strength=0.5, frac=0.2))
    assert calls["strength"] == 0.5 and calls["mask"][:, w // 2].min() == 255 and calls["mask"][:, 0].max() == 0
    assert (out[:, 0] == (255, 0, 0)).all() and (out[:, -1] == (255, 0, 0)).all()   # 경계는 새로 그림
    assert (out[:, 30:170] == 40).all()                                             # 띠 밖은 원본
