"""파노라마 생성 엔진 (staged loading, 백엔드 공통).

1) 텍스트 인코더만 올려 모든 프롬프트를 임베딩 → 해제
2) 트랜스포머를 프로필에 맞게 적재(bf16/fp8/nf4/gguf) + 파노라마 LoRA (+ 선택: 속도 LoRA)
3) 임베딩으로 2048×1024 생성 (일반 이미지 pano=False 면 파노라마 LoRA 없이 지정 크기)

백엔드별 차이(임베딩 종류, CFG 방식)는 _encode_one / _call_kwargs 에만 있다. 백엔드 목록은 backends.py.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from panogen.backends import CLASSES, resolve_backend, sampling
from panogen.device import free_memory
from panogen.profiles import Profile

Log = Callable[[str], None]


@dataclass
class Job:
    prompt: str
    seed: int
    full_prompt: str = ""


def build_prompt(prompt: str, prefix: str) -> str:
    p = prompt.strip()
    if prefix and not p.lower().startswith(prefix.strip().lower()[:24]):
        p = prefix + p[0].lower() + p[1:] if p else prefix
    return p


class Engine:
    def __init__(self, root: Path, cfg: dict, profile: Profile, bf16: bool = True, log: Log = print,
                 backend: str | None = None, fast: bool = False, pano: bool = True):
        import torch

        self.torch = torch
        self.root = root
        self.cfg = cfg
        self.profile = profile
        self.log = log
        self.b = resolve_backend(cfg, backend)
        self.pano = pano          # False: 일반 이미지 — 파노라마 LoRA·순환 패딩 없이 같은 베이스 모델
        self.fast = fast and bool(self.b.get("speed_lora"))
        self.sampling = sampling(self.b, cfg.get("generate", {}), self.fast)
        self.dtype = torch.bfloat16 if (bf16 or profile.device == "mps") else torch.float16
        self.Pipeline = self._pipeline_class()
        self.pipe = None

    def _pipeline_class(self):
        import diffusers

        if self.b["kind"] == "dit360":
            if not self.pano:      # 일반 이미지: 순환 패딩 없는 FLUX.1-dev 그대로
                return diffusers.FluxPipeline
            from panogen.dit360 import load_pipeline_class

            return load_pipeline_class(self.root)

        return getattr(diffusers, CLASSES[self.b["kind"]][0])

    @property
    def max_len(self) -> int:
        return int(self.cfg.get("generate", {}).get("max_sequence_length") or self.b["max_sequence_length"])

    # ---------- 1. 텍스트 ----------
    def _encode_one(self, enc, prompt: str, dev) -> dict:
        kind = self.b["kind"]

        def cpu(x):   # 일부 버전은 mask 를 None 으로 돌려줌 (전부 유효 토큰일 때)
            if x is None:
                return None
            return x.to("cpu", self.dtype) if x.is_floating_point() else x.to("cpu")

        if kind == "dit360":
            pe, pooled, _ = enc.encode_prompt(prompt=prompt, prompt_2=None, device=dev, max_sequence_length=self.max_len)
            return {"prompt_embeds": cpu(pe), "pooled_prompt_embeds": cpu(pooled)}
        neg = self.b.get("negative_prompt") or " "
        if kind == "qwen":
            pe, mask = enc.encode_prompt(prompt=prompt, device=dev, max_sequence_length=self.max_len)
            out = {"prompt_embeds": cpu(pe), "prompt_embeds_mask": cpu(mask)}
            if self.sampling["true_cfg"] > 1:
                npe, nmask = enc.encode_prompt(prompt=neg, device=dev, max_sequence_length=self.max_len)
                out.update(negative_prompt_embeds=cpu(npe), negative_prompt_embeds_mask=cpu(nmask))
            return {k: v for k, v in out.items() if v is not None}
        pe, _ = enc.encode_prompt(prompt=prompt, device=dev, max_sequence_length=self.max_len)   # klein
        out = {"prompt_embeds": cpu(pe)}
        if self.sampling["guidance"] > 1:
            npe, _ = enc.encode_prompt(prompt=neg, device=dev, max_sequence_length=self.max_len)
            out["negative_prompt_embeds"] = cpu(npe)
        return out

    def encode(self, prompts: list[str]) -> list[dict]:
        t = self.torch
        dev = self.profile.text_device
        self.log(f"[1/3] 텍스트 인코더 적재 ({dev}) — {self.b['name']} …")
        t0 = time.time()
        enc = self.Pipeline.from_pretrained(self.b["base"], transformer=None, vae=None, torch_dtype=self.dtype)
        enc.to(dev)
        with t.inference_mode():
            out = [self._encode_one(enc, p, t.device(dev)) for p in prompts]
        del enc
        free_memory()
        self.log(f"      {len(prompts)}개 인코딩 완료 ({time.time() - t0:.0f}s), 텍스트 인코더 해제")
        return out

    # ---------- 2. 트랜스포머 ----------
    def _load_transformer(self):
        import diffusers

        t = self.torch
        b = self.b
        Tr = getattr(diffusers, CLASSES[b["kind"]][1])
        kind = self.profile.transformer
        if kind in ("bf16", "fp8"):
            return Tr.from_pretrained(b["base"], subfolder="transformer", torch_dtype=self.dtype)
        if kind == "nf4":
            q = diffusers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=self.dtype)
            return Tr.from_pretrained(b["base"], subfolder="transformer", quantization_config=q, torch_dtype=self.dtype)
        if kind == "gguf":
            from huggingface_hub import hf_hub_download

            if not b["gguf_repo"]:
                raise RuntimeError(f"백엔드 {b['name']} 에 gguf_repo/gguf_file 설정이 없습니다.")
            path = hf_hub_download(b["gguf_repo"], b["gguf_file"])
            return Tr.from_single_file(path, quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=self.dtype),
                                       config=b["base"], subfolder="transformer", torch_dtype=self.dtype)
        raise ValueError(kind)

    def _load_loras(self, pipe):
        b = self.b
        names, weights = [], []
        for key, scale, adapter in (("lora", b["lora_scale"], "pano"), ("speed_lora", 1.0, "speed")):
            if not b.get(key) or (adapter == "speed" and not self.fast) or (adapter == "pano" and not self.pano):
                continue
            kw = {"weight_name": b[f"{key}_file"]} if b.get(f"{key}_file") else {}
            pipe.load_lora_weights(b[key], adapter_name=adapter, **kw)
            names.append(adapter)
            weights.append(float(scale))
        if names:
            pipe.set_adapters(names, adapter_weights=weights)
        return names

    def load(self):
        if self.pipe is not None:
            return self.pipe
        t = self.torch
        p = self.profile
        self.log(f"[2/3] 트랜스포머 적재 — {self.b['name']} · 프로필 {p.name} ({p.transformer}) …")
        t0 = time.time()
        tr = self._load_transformer()
        empty = {"text_encoder": None, "tokenizer": None}
        if self.b["kind"] == "dit360":
            empty.update(text_encoder_2=None, tokenizer_2=None)
        pipe = self.Pipeline.from_pretrained(self.b["base"], transformer=tr, torch_dtype=self.dtype, **empty)
        try:
            adapters = self._load_loras(pipe)
        except Exception as e:  # GGUF + LoRA 등 조합이 diffusers 버전에 따라 막힐 수 있음
            raise RuntimeError(f"LoRA 적용 실패 ({type(e).__name__}: {e}). "
                               "diffusers 를 최신으로 올리거나 다른 프로필을 사용하세요.") from e

        if p.transformer == "fp8":
            # LoRA 를 bf16 가중치에 병합한 뒤 fp8 로 저장 → VRAM 절반
            if adapters:
                pipe.fuse_lora()
                pipe.unload_lora_weights()
            pipe.transformer.enable_layerwise_casting(storage_dtype=t.float8_e4m3fn, compute_dtype=self.dtype)

        if p.transformer == "nf4":
            pipe.vae.to(p.device)      # bnb 4bit 트랜스포머는 적재 시 이미 GPU 에 있음
        else:
            pipe.to(p.device)
        if p.vae_tiling and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=False)
        self.pipe = pipe
        self.log(f"      준비 완료 ({time.time() - t0:.0f}s)")
        return pipe

    # ---------- 3. 생성 ----------
    def _call_kwargs(self) -> dict:
        s = self.sampling
        if self.b["kind"] == "qwen":
            return {"true_cfg_scale": s["true_cfg"]}
        kw = {"guidance_scale": s["guidance"]}
        if self.b["kind"] == "dit360":
            kw["max_sequence_length"] = self.max_len
        return kw

    def generate(self, emb: dict, seed: int, init=None, size: tuple[int, int] | None = None):
        """init: imagein.InitImage — 첨부 이미지를 프롬프트대로 고쳐서 (None 이면 텍스트만).
        size: (가로, 세로). 없으면 설정의 파노라마 크기(2048×1024)."""
        t = self.torch
        g = self.cfg["generate"]
        pipe = self.load()
        dev = self.profile.device
        w, h = size or (int(g["width"]), int(g["height"]))
        kw = dict(**{k: v.to(dev) for k, v in emb.items()}, width=w, height=h,
                  num_inference_steps=self.sampling["steps"], **self._call_kwargs())
        gen = lambda s: t.Generator("cpu").manual_seed(s)
        self.circular = self.b["kind"] == "dit360" and self.pano and init is None   # DiT360 순환 패딩은 360 텍스트 생성에만
        with t.inference_mode():
            if init is None:
                img = self._run(pipe, **kw, generator=gen(seed))
            else:
                img = self._generate_from_image(init, kw, gen, seed)
        free_memory()
        return img

    def _run(self, pipe, **kw):
        """파이프라인 1회 실행 → PIL 이미지. 맥이면 디코딩 직전에 트랜스포머를 내림 (_drop_before_decode)."""
        if self.profile.device == "mps":
            return self._drop_before_decode(pipe, kw)
        return pipe(**kw).images[0]

    def _drop_before_decode(self, pipe, kw: dict):
        """맥(통합 메모리) 전용: 디노이즈가 끝나고 VAE 디코딩 직전에 트랜스포머를 메모리에서 내린다.
        16GB 에서 Qwen 20B(Q4 ~12GB)는 트랜스포머 + 디코딩 활성값이 겹치는 순간 메모리가 넘쳤음.
        다음 생성·이음새 인페인팅 때는 load() 가 트랜스포머를 다시 올린다.
        (refine 은 타일마다 호출하므로 쓰지 않음)"""
        vae = pipe.vae
        orig = vae.decode

        def decode(*a, **k):
            vae.decode = orig
            pipe.transformer = None
            self.pipe = None
            free_memory()
            self.log("      (메모리 절약) 디코딩 전에 트랜스포머 해제")
            return orig(*a, **k)

        vae.decode = decode
        try:
            return pipe(**kw).images[0]
        finally:
            vae.decode = orig

    def _generate_from_image(self, init, kw: dict, gen, seed: int):
        import numpy as np
        from PIL import Image

        kind = self.b["kind"]
        if kind == "klein":    # FLUX.2 klein: 첨부 전부를 참조 이미지로 편집 (360 은 "Convert to an Equirectangular …")
            refs = [Image.fromarray(s) for s in init.sources]
            self.log(f"      이미지 입력: 참조 이미지 {len(refs)}장 편집 (klein)")
            return self._run(self.pipe, image=refs, **kw, generator=gen(seed))
        if init.unused:
            self.log(f"      (추가 첨부 {init.unused}장은 이 모드에서 쓰지 않음 — 여러 장 참조는 klein360)")
        if init.mode == "img2img":
            self.log(f"      이미지 입력: {'파노라마' if self.pano else '이미지'} 수정 (strength {init.strength})")
            return self._run(self.sibling(IMG2IMG[kind]), image=Image.fromarray(init.image), strength=init.strength,
                             **kw, generator=gen(seed))
        # 일반 사진: 정면에 투영된 사진 밖을 새로 그린 뒤, 사진 영역은 원본으로 다시 합성
        where = "정면" if len(init.yaws or [0]) == 1 else "·".join(f"{y:.0f}°" for y in init.yaws)
        self.log(f"      이미지 입력: 사진 {len(init.yaws or [0])}장 → 360° 확장 (화각 {init.fov:.0f}°, 배치 {where})")
        out = self._run(self.sibling(OUTPAINT[kind]), image=Image.fromarray(init.image),
                        mask_image=Image.fromarray(init.mask), strength=1.0, **kw, generator=gen(seed))
        soft = _feather(init.mask, 6)[..., None]
        arr = np.asarray(out.convert("RGB"), np.float32) * soft + init.image.astype(np.float32) * (1 - soft)
        img = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
        if init.strength > 0:   # 사진 영역까지 프롬프트를 반영하고 경계를 자연스럽게
            self.log(f"      프롬프트 반영 (strength {init.strength})")
            img = self._run(self.sibling(IMG2IMG[kind]), image=img, strength=init.strength, **kw,
                            generator=gen(seed + 7))
        return img

    def sibling(self, cls_name: str):
        """적재된 모듈(트랜스포머·VAE·스케줄러)을 그대로 공유하는 다른 파이프라인 (img2img·inpaint).
        from_pipe 는 기본으로 전체를 float32 로 바꿔서(양자화 트랜스포머 포함) 쓰지 않는다."""
        import inspect

        import diffusers

        P = getattr(diffusers, cls_name)
        params = inspect.signature(P.__init__).parameters
        sib = P(**{k: v for k, v in self.load().components.items() if k in params})
        sib.set_progress_bar_config(disable=False)
        return sib

    # ---------- 이음새 인페인팅 ----------
    @property
    def can_inpaint(self) -> bool:
        # dit360 텍스트 생성은 추론 중 순환 패딩이 있어 불필요, klein 은 마스크 인페인트 파이프라인이 없음
        return self.b["kind"] in INPAINT and not getattr(self, "circular", False)

    def inpaint_seam(self, img, emb: dict, seed: int, strength: float = 0.5, frac: float = 0.10):
        """가로 50% 이동해 0°/360° 경계를 가운데로 → 가운데 띠만 같은 모델로 다시 그림 → 되돌림.
        띠 밖은 원본을 그대로 쓰도록 부드러운 마스크로 합성 (VAE 왕복 손실 방지)."""
        import numpy as np
        from PIL import Image

        from panogen import seam

        t = self.torch
        arr = np.asarray(img.convert("RGB"))
        h, w = arr.shape[:2]
        rolled = seam.roll(arr)
        half = max(8, int(w * frac / 2))
        hard = np.zeros((h, w), np.uint8)
        hard[:, w // 2 - half: w // 2 + half] = 255
        ip = self.sibling(INPAINT[self.b["kind"]])
        with t.inference_mode():
            out = self._run(ip, **{k: v.to(self.profile.device) for k, v in emb.items()},
                     image=Image.fromarray(rolled), mask_image=Image.fromarray(hard), strength=strength,
                     width=w, height=h, num_inference_steps=self.sampling["steps"], **self._call_kwargs(),
                     generator=t.Generator("cpu").manual_seed(seed + 1))
        del ip
        free_memory()
        # 띠 가장자리 1/3 구간에서 원본 ↔ 인페인팅본을 선형으로 섞음
        x = np.abs(np.arange(w) - w / 2)
        soft = np.clip((half - x) / max(1, half / 3), 0, 1).astype(np.float32)[None, :, None]
        mixed = rolled * (1 - soft) + np.asarray(out.convert("RGB"), np.float32) * soft
        return Image.fromarray(seam.unroll(np.clip(np.rint(mixed), 0, 255).astype(np.uint8)))


INPAINT = {"qwen": "QwenImageInpaintPipeline", "dit360": "FluxInpaintPipeline"}
OUTPAINT = INPAINT
IMG2IMG = {"qwen": "QwenImageImg2ImgPipeline", "dit360": "FluxImg2ImgPipeline"}


def _feather(mask, r: int):
    """255/0 마스크 → 0~1, 경계를 r 픽셀 박스 블러로 부드럽게."""
    import numpy as np

    m = mask.astype(np.float32) / 255
    if r <= 0:
        return m
    k = np.ones(2 * r + 1, np.float32) / (2 * r + 1)
    m = np.apply_along_axis(lambda x: np.convolve(np.pad(x, r, mode="edge"), k, "valid"), 1, m)
    return np.apply_along_axis(lambda x: np.convolve(np.pad(x, r, mode="edge"), k, "valid"), 0, m)


def make_jobs(prompts: list[str], seed: int, count: int, prefix: str) -> list[Job]:
    jobs = []
    for p in prompts:
        for i in range(count):
            s = (seed + i) if seed >= 0 else random.randint(0, 2**31 - 1)
            jobs.append(Job(p, s, build_prompt(p, prefix)))
    return jobs
