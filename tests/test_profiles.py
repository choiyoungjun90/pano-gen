import pytest

from panogen.profiles import auto_profile, resolve


@pytest.mark.parametrize("vram,expect", [(80, "cuda-full"), (32, "cuda-full"), (24, "cuda-24g"),
                                         (16, "cuda-12g"), (12, "cuda-12g")])
def test_auto_cuda(vram, expect):
    assert auto_profile("cuda", vram).name == expect


def test_auto_mps_and_cpu():
    assert auto_profile("mps", None).name == "mps-16g"
    with pytest.raises(RuntimeError):
        auto_profile("cpu", None)


def test_resolve_device_mismatch():
    with pytest.raises(RuntimeError):
        resolve("mps-16g", "cuda", 24)
