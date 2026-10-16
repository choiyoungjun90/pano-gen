"""모델 백엔드 — 같은 staged 생성 흐름에 넣을 파노라마 모델 조합.

| 백엔드    | 베이스                       | 파노라마 LoRA                     | 비고                                   |
|-----------|------------------------------|-----------------------------------|----------------------------------------|
| dit360    | FLUX.1-dev (12B)             | DiT360                            | 추론 중 좌우 순환 패딩 (third_party)   |
| qwen360   | Qwen-Image-2512 (20B)        | Qwen 360 Diffusion v2             | true CFG, Lightning LoRA 로 8 step 가능 |
| klein360  | FLUX.2 [klein] base 9B       | crafiq 360 panorama               | CFG (base 모델)                        |

설정은 config 의 [backends.<이름>] 에 있고 [model] backend 로 고른다.
[model] 에 base/lora 등을 직접 적으면 선택된 백엔드 값을 덮어쓴다 (테스트·실험용).
"""

from __future__ import annotations

KINDS = ("dit360", "qwen", "klein")

# pipeline kind → (diffusers 파이프라인 클래스, 트랜스포머 클래스). dit360 은 third_party 의 DiT360Pipeline.
CLASSES = {
    "dit360": (None, "FluxTransformer2DModel"),
    "qwen": ("QwenImagePipeline", "QwenImageTransformer2DModel"),
    "klein": ("Flux2KleinPipeline", "Flux2Transformer2DModel"),
}

DEFAULTS = {"kind": "dit360", "lora": "", "lora_file": "", "lora_scale": 1.0, "speed_lora": "",
            "speed_lora_file": "", "speed_steps": 8, "gguf_repo": "", "gguf_file": "", "prompt_prefix": "",
            "negative_prompt": "", "steps": 28, "guidance": 3.5, "true_cfg": 1.0, "max_sequence_length": 512}


def resolve_backend(cfg: dict, name: str | None = None) -> dict:
    m = dict(cfg.get("model", {}))
    name = name or m.pop("backend", None) or "dit360"
    m.pop("backend", None)
    specs = cfg.get("backends", {})
    if specs and name not in specs:
        raise ValueError(f"알 수 없는 백엔드: {name} ({', '.join(specs)})")
    b = {**DEFAULTS, **specs.get(name, {}), **m, "name": name}
    if b["kind"] not in KINDS:
        raise ValueError(f"백엔드 {name} 의 kind 가 잘못됨: {b['kind']}")
    if not b.get("base"):
        raise ValueError(f"백엔드 {name} 에 base 모델이 없습니다.")
    return b


def sampling(b: dict, g: dict, fast: bool = False) -> dict:
    """steps/guidance 결정: 명시한 [generate] 값 > 백엔드 기본값. fast 면 speed LoRA 기준 step·CFG 끔."""
    steps = g.get("steps") or b["steps"]
    if fast and b.get("speed_lora"):
        return {"steps": int(g.get("steps") or b["speed_steps"]), "guidance": 1.0, "true_cfg": 1.0}
    return {"steps": int(steps), "guidance": float(g.get("guidance") or b["guidance"]),
            "true_cfg": float(b["true_cfg"])}
