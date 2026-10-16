"""장치·메모리 감지 (맥/윈도우/리눅스 공통)."""

from __future__ import annotations

import platform


def detect() -> dict:
    info = {"os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "python": platform.python_version(), "device": "cpu", "vram_gb": None,
            "gpu": None, "ram_gb": None, "bf16": False}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info
    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(device="cuda", gpu=p.name, vram_gb=round(p.total_memory / 1024**3, 1),
                    bf16=torch.cuda.is_bf16_supported())
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info.update(device="mps", gpu="Apple Silicon (통합 메모리)", bf16=True)
    return info


def free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def pick_device() -> str:
    return detect()["device"]


def peak_memory_gb() -> float | None:
    """프로세스 최대 메모리(GB). 맥은 MPS 드라이버 할당량과 RSS 최대치 중 큰 값."""
    import resource
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gb = rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2
    try:
        import torch
        if torch.cuda.is_available():
            gb = max(gb, torch.cuda.max_memory_allocated() / 1024**3)
        elif torch.backends.mps.is_available():
            gb = max(gb, torch.mps.driver_allocated_memory() / 1024**3)
    except Exception:
        pass
    return round(gb, 2)
