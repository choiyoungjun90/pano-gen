"""하드웨어 프로필 — 같은 DiT360 파이프라인을 메모리 크기에 맞춰 다르게 적재.

| 프로필    | 대상                         | 트랜스포머 적재           | 텍스트 인코더 |
|-----------|------------------------------|---------------------------|---------------|
| cuda-full | NVIDIA VRAM 32GB 이상        | bf16 그대로 (~24GB)       | GPU           |
| cuda-24g  | NVIDIA VRAM 20~24GB          | LoRA 병합 후 fp8 저장(~12GB), bf16 연산 | GPU |
| cuda-12g  | NVIDIA VRAM 12~16GB          | 4bit NF4 (~7GB)           | CPU           |
| mps-16g   | Apple Silicon 16GB (실험적)  | 4bit GGUF (~7GB)          | CPU           |

공식 DiT360 추론은 최적화 없이 약 37GB 를 쓰므로, 모든 프로필은 '텍스트 인코딩 → 해제 →
트랜스포머 적재' 순서(staged)로 동시에 메모리에 올라가는 양을 줄인다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    device: str             # cuda | mps
    transformer: str        # bf16 | fp8 | nf4 | gguf
    text_device: str        # 텍스트 인코더 실행 장치
    vae_tiling: bool
    min_ram_gb: int         # 권장 시스템 RAM
    note: str


PROFILES = {
    "cuda-full": Profile("cuda-full", "cuda", "bf16", "cuda", False, 32, "VRAM 32GB 이상"),
    "cuda-24g": Profile("cuda-24g", "cuda", "fp8", "cuda", False, 32, "VRAM 20~24GB (RTX 3090/4090)"),
    "cuda-12g": Profile("cuda-12g", "cuda", "nf4", "cpu", True, 32, "VRAM 12~16GB"),
    "mps-16g": Profile("mps-16g", "mps", "gguf", "cpu", True, 16, "Apple Silicon 16GB, 실험적·느림"),
}


def auto_profile(device: str, vram_gb: float | None) -> Profile:
    if device == "cuda" and vram_gb:
        if vram_gb >= 30:
            return PROFILES["cuda-full"]
        if vram_gb >= 20:
            return PROFILES["cuda-24g"]
        return PROFILES["cuda-12g"]
    if device == "mps":
        return PROFILES["mps-16g"]
    raise RuntimeError("NVIDIA GPU(CUDA) 또는 Apple Silicon(MPS)이 필요합니다. CPU 생성은 지원하지 않습니다.")


def resolve(name: str, device: str, vram_gb: float | None) -> Profile:
    if name == "auto":
        return auto_profile(device, vram_gb)
    if name not in PROFILES:
        raise ValueError(f"알 수 없는 프로필: {name} ({', '.join(PROFILES)})")
    p = PROFILES[name]
    if p.device != device:
        raise RuntimeError(f"프로필 {name} 은 {p.device} 장치용인데 현재 장치는 {device} 입니다.")
    return p
