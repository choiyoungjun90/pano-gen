"""작은 무작위 FLUX 구성요소로 DiT360 staged 생성 경로 전체를 CPU 에서 검증.

실제 모델 없이: 텍스트 인코딩 → 해제 → 트랜스포머+LoRA 적재(bf16, fp8 병합) → 순환 패딩 생성 → 디코딩.
torch/diffusers 나 third_party/DiT360 이 없으면 건너뜀.
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "third_party" / "DiT360" / "src" / "pipeline.py").exists():
    pytest.skip("third_party/DiT360 없음 (scripts/fetch_dit360.py)", allow_module_level=True)


def _tokenizer(max_len):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<unk>": 1, "</s>": 2, **{w: i + 3 for i, w in enumerate(
        "this is a panorama the image shows misty temple courtyard at dawn".split())}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", unk_token="<unk>",
                                   eos_token="</s>", model_max_length=max_len)


@pytest.fixture(scope="module")
def tiny_repo(tmp_path_factory):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    from peft import LoraConfig, get_peft_model_state_dict
    from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel

    from panogen.dit360 import load_pipeline_class

    torch.manual_seed(0)
    Pipe = load_pipeline_class(ROOT)
    tr = FluxTransformer2DModel(patch_size=1, in_channels=16, num_layers=1, num_single_layers=1,
                                attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
                                pooled_projection_dim=32, guidance_embeds=True, axes_dims_rope=(4, 6, 6))
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(4, 4, 4, 4),
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                        layers_per_block=1, norm_num_groups=2, shift_factor=0.0, scaling_factor=1.0)
    clip = CLIPTextModel(CLIPTextConfig(vocab_size=64, hidden_size=32, intermediate_size=37, num_hidden_layers=1,
                                        num_attention_heads=2, max_position_embeddings=77, projection_dim=32))
    t5 = T5EncoderModel(T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=37, num_layers=1, num_heads=2))
    sched = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
                                            base_image_seq_len=256, max_image_seq_len=4096)
    pipe = Pipe(scheduler=sched, vae=vae, text_encoder=clip, tokenizer=_tokenizer(77),
                text_encoder_2=t5, tokenizer_2=_tokenizer(64), transformer=tr)
    base = tmp_path_factory.mktemp("tiny_flux")
    pipe.save_pretrained(base)

    tr.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights=False,
                              target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    lora = tmp_path_factory.mktemp("tiny_lora")
    Pipe.save_lora_weights(lora, transformer_lora_layers=get_peft_model_state_dict(tr))
    return base, lora


def _cfg(base, lora):
    return {"model": {"base": str(base), "lora": str(lora), "lora_scale": 1.0},
            "generate": {"width": 128, "height": 64, "steps": 2, "guidance": 2.8, "max_sequence_length": 64}}


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_staged_generation(tiny_repo, kind):
    from panogen.engine import Engine, build_prompt
    from panogen.profiles import Profile

    base, lora = tiny_repo
    prof = Profile("cpu-test", "cpu", kind, "cpu", False, 0, "test")
    eng = Engine(ROOT, _cfg(base, lora), prof, bf16=True, log=lambda m: None)
    prompt = build_prompt("misty temple courtyard at dawn", "This is a panorama. The image shows ")
    [emb] = eng.encode([prompt])
    img = eng.generate(emb, seed=7)
    assert img.size == (128, 64)
    if kind == "fp8":
        assert next(eng.pipe.transformer.parameters()).dtype == torch.float8_e4m3fn
    else:
        assert "pano" in eng.pipe.get_active_adapters()


def test_lora_changes_output(tiny_repo):
    import numpy as np

    from panogen.engine import Engine
    from panogen.profiles import Profile

    base, lora = tiny_repo
    outs = []
    for scale in (0.0, 1.0):
        cfg = _cfg(base, lora)
        cfg["model"]["lora_scale"] = scale
        eng = Engine(ROOT, cfg, Profile("t", "cpu", "bf16", "cpu", False, 0, ""), log=lambda m: None)
        [emb] = eng.encode(["This is a panorama. The image shows a temple"])
        outs.append(np.asarray(eng.generate(emb, 5)).astype(int))
    assert np.abs(outs[0] - outs[1]).mean() > 0.5


def test_same_seed_same_image(tiny_repo):
    import numpy as np

    from panogen.engine import Engine
    from panogen.profiles import Profile

    base, lora = tiny_repo
    eng = Engine(ROOT, _cfg(base, lora), Profile("t", "cpu", "bf16", "cpu", False, 0, ""), log=lambda m: None)
    [emb] = eng.encode(["This is a panorama. The image shows a temple"])
    a, b = np.asarray(eng.generate(emb, 3)), np.asarray(eng.generate(emb, 3))
    assert np.array_equal(a, b)


def test_prompt_prefix():
    from panogen.engine import build_prompt

    assert build_prompt("A misty temple", "This is a panorama. The image shows ") == \
        "This is a panorama. The image shows a misty temple"
    assert build_prompt("This is a panorama. foo", "This is a panorama. ") == "This is a panorama. foo"


def test_cli_generate_end_to_end(tiny_repo, tmp_path, monkeypatch):
    """CLI: 프롬프트 파일 2줄 × 2장 → 2K + 업스케일(Lanczos) + meta.json."""
    import json

    import panogen.device as dev
    import panogen.profiles as profs
    from panogen.cli import main
    from panogen.profiles import Profile

    base, lora = tiny_repo
    monkeypatch.setattr(dev, "detect", lambda: {"device": "cpu", "vram_gb": None, "gpu": "test", "bf16": True})
    monkeypatch.setattr(profs, "resolve", lambda *a: Profile("cpu-test", "cpu", "bf16", "cpu", False, 0, ""))
    out = tmp_path / "out"
    conf = tmp_path / "t.toml"
    conf.write_text(
        f'[model]\nbase = "{base.as_posix()}"\nlora = "{lora.as_posix()}"\n'
        f'[generate]\nwidth = 128\nheight = 64\nsteps = 2\nmax_sequence_length = 64\n'
        f'[upscale]\nmodel_path = ""\ntarget_width = 256\n[output]\ndir = "{out.as_posix()}"\nsave_png = false\n',
        encoding="utf-8")
    prompts = tmp_path / "p.txt"
    prompts.write_text("# 주석\nmisty temple courtyard\n안개 낀 사찰 마당\n", encoding="utf-8")
    assert main(["--config", str(conf), "generate", "--prompts", str(prompts), "-c", "2", "--seed", "10",
                 "-n", "batch"]) == 0
    batch = out / "batch"
    # 기본은 2K 까지만 — 확인 후 고른 시드만 업스케일
    files = sorted(p.name for p in batch.iterdir())
    assert files == sorted(f"p{p:02d}_seed{s}{ext}" for p in (1, 2) for s in (10, 11) for ext in (".json", "_2k.png"))

    assert main(["--config", str(conf), "upscale", str(batch)]) == 1            # 시드 미지정 → 거부
    assert main(["--config", str(conf), "upscale", str(batch), "--seed", "99"]) == 1
    assert main(["--config", str(conf), "upscale", str(batch), "--seed", "11"]) == 0
    assert sorted(p.name for p in batch.glob("*.jpg")) == ["p01_seed11_256px.jpg", "p02_seed11_256px.jpg"]
    metas = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in batch.glob("*.json")}
    assert metas["p01_seed11"]["output"] == "p01_seed11_256px.jpg" and "upscale_seconds" in metas["p01_seed11"]
    assert "output" not in metas["p01_seed10"]

    # --upscale 이면 예전처럼 생성 직후 8K 까지
    assert main(["--config", str(conf), "generate", "-p", "misty temple", "--seed", "5", "-n", "one",
                 "--upscale"]) == 0
    assert sorted(p.name for p in (out / "one").iterdir()) == ["seed5.json", "seed5_256px.jpg", "seed5_2k.png"]


def test_cli_bench(tiny_repo, tmp_path, monkeypatch):
    """bench: 백엔드별 폴더에 프롬프트×시드 2K 생성, 잘못된 백엔드는 건너뛰고 계속."""
    import json

    import panogen.device as dev
    import panogen.profiles as profs
    from panogen.cli import main
    from panogen.profiles import Profile

    base, lora = tiny_repo
    monkeypatch.setattr(dev, "detect", lambda: {"device": "cpu", "vram_gb": None, "gpu": "test", "bf16": True})
    monkeypatch.setattr(profs, "resolve", lambda *a: Profile("cpu-test", "cpu", "bf16", "cpu", False, 0, ""))
    out = tmp_path / "out"
    conf = tmp_path / "t.toml"
    conf.write_text(
        f'[backends.dit360]\nbase = "{base.as_posix()}"\nlora = "{lora.as_posix()}"\nsteps = 2\n'
        f'[generate]\nwidth = 128\nheight = 64\nmax_sequence_length = 64\n'
        f'[output]\ndir = "{out.as_posix()}"\n', encoding="utf-8")
    prompts = tmp_path / "p.txt"
    prompts.write_text("misty temple\nsnowy temple\n", encoding="utf-8")
    assert main(["--config", str(conf), "bench", "--backends", "dit360,nope", "--set", str(prompts),
                 "--seeds", "3", "9", "--tag", "t"]) == 0
    d = out / "bench_t_dit360"
    assert sorted(p.name for p in d.glob("*_2k.png")) == \
        ["p01_seed3_2k.png", "p01_seed9_2k.png", "p02_seed3_2k.png", "p02_seed9_2k.png"]
    meta = json.loads((d / "p01_seed3.json").read_text(encoding="utf-8"))
    assert meta["backend"] == "dit360" and meta["steps"] == 2 and "peak_memory_gb" in meta
    assert not (out / "bench_t_nope").exists() or not list((out / "bench_t_nope").glob("*.png"))
    summ = json.loads((out / "bench_t.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in summ["runs"]] == ["done", "failed"] and summ["seeds"] == [3, 9]
    assert summ["prompts"] == ["misty temple", "snowy temple"]


def test_cli_upscale_refine(tiny_repo, tmp_path, monkeypatch):
    """upscale --refine: 생성 기록의 프롬프트로 같은 모델 타일 img2img 후 업스케일, json 에 기록."""
    import json

    import panogen.device as dev
    import panogen.profiles as profs
    from panogen.cli import main
    from panogen.profiles import Profile

    base, lora = tiny_repo
    monkeypatch.setattr(dev, "detect", lambda: {"device": "cpu", "vram_gb": None, "gpu": "test", "bf16": True})
    monkeypatch.setattr(profs, "resolve", lambda *a: Profile("cpu-test", "cpu", "bf16", "cpu", False, 0, ""))
    out = tmp_path / "out"
    conf = tmp_path / "t.toml"
    conf.write_text(
        f'[backends.dit360]\nbase = "{base.as_posix()}"\nlora = "{lora.as_posix()}"\nsteps = 2\n'
        f'[generate]\nwidth = 128\nheight = 64\nmax_sequence_length = 64\n'
        f'[upscale]\nmodel_path = ""\ntarget_width = 512\n[output]\ndir = "{out.as_posix()}"\nsave_png = false\n',
        encoding="utf-8")
    assert main(["--config", str(conf), "generate", "-p", "misty temple", "--seed", "4", "-n", "r"]) == 0
    assert main(["--config", str(conf), "upscale", str(out / "r"), "--seed", "4", "--refine", "--strength", "0.5"]) == 0
    meta = json.loads((out / "r" / "seed4.json").read_text(encoding="utf-8"))
    assert meta["refined"] and meta["refine_strength"] == 0.5 and meta["output"] == "seed4_512px.jpg"


@pytest.mark.parametrize("shape,mode", [((48, 64), "outpaint"), ((64, 128), "img2img")])
def test_cli_generate_with_image(tiny_repo, tmp_path, monkeypatch, shape, mode):
    """--image: 일반 사진은 360° 확장, 2:1 파노라마는 수정. 원본은 결과 폴더에 input1.* 로 보관."""
    import json

    import numpy as np
    from PIL import Image

    import panogen.device as dev
    import panogen.profiles as profs
    from panogen.cli import main
    from panogen.profiles import Profile

    base, lora = tiny_repo
    monkeypatch.setattr(dev, "detect", lambda: {"device": "cpu", "vram_gb": None, "gpu": "test", "bf16": True})
    monkeypatch.setattr(profs, "resolve", lambda *a: Profile("cpu-test", "cpu", "bf16", "cpu", False, 0, ""))
    out = tmp_path / "out"
    conf = tmp_path / "t.toml"
    conf.write_text(
        f'[backends.dit360]\nbase = "{base.as_posix()}"\nlora = "{lora.as_posix()}"\nsteps = 2\n'
        f'[generate]\nwidth = 128\nheight = 64\nmax_sequence_length = 64\n'
        f'[seam]\ninpaint = "off"\n[output]\ndir = "{out.as_posix()}"\n', encoding="utf-8")
    src = tmp_path / "photo.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (*shape, 3), dtype=np.uint8)).save(src)
    assert main(["--config", str(conf), "generate", "-p", "misty temple", "--seed", "2", "-n", "img",
                 "--image", str(src), "--strength", "0.5"]) == 0
    d = out / "img"
    assert (d / "seed2_2k.png").exists() and (d / "input1.jpg").exists()
    meta = json.loads((d / "seed2.json").read_text(encoding="utf-8"))
    assert meta["input"]["mode"] == mode and meta["input"]["strength"] == 0.5


def _cli_env(tiny_repo, tmp_path, monkeypatch, seam="off"):
    import panogen.device as dev
    import panogen.profiles as profs
    from panogen.profiles import Profile

    base, lora = tiny_repo
    monkeypatch.setattr(dev, "detect", lambda: {"device": "cpu", "vram_gb": None, "gpu": "test", "bf16": True})
    monkeypatch.setattr(profs, "resolve", lambda *a: Profile("cpu-test", "cpu", "bf16", "cpu", False, 0, ""))
    out = tmp_path / "out"
    conf = tmp_path / "t.toml"
    conf.write_text(
        f'[backends.dit360]\nbase = "{base.as_posix()}"\nlora = "{lora.as_posix()}"\nsteps = 2\n'
        f'[generate]\nwidth = 128\nheight = 64\nmax_sequence_length = 64\nflat_base = 128\n'
        f'[seam]\ninpaint = "{seam}"\n[upscale]\nmodel_path = ""\ntarget_width = 256\n'
        f'[output]\ndir = "{out.as_posix()}"\nsave_png = false\n', encoding="utf-8")
    return out, conf


def _photo(path, shape, seed=0):
    import numpy as np
    from PIL import Image

    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (*shape, 3), dtype=np.uint8)).save(path)
    return str(path)


def test_cli_three_photos_spread_around(tiny_repo, tmp_path, monkeypatch):
    """사진 3장 → 0°·120°·240° 에 배치해 360° 확장, 원본 3장 보관."""
    import json

    from panogen.cli import main

    out, conf = _cli_env(tiny_repo, tmp_path, monkeypatch)
    imgs = [_photo(tmp_path / f"p{i}.jpg", (60, 80), i) for i in range(3)]
    args = ["--config", str(conf), "generate", "-p", "misty temple", "--seed", "1", "-n", "three"]
    for im in imgs:
        args += ["--image", im]
    assert main(args) == 0
    d = out / "three"
    assert all((d / f"input{k}.jpg").exists() for k in (1, 2, 3))
    meta = json.loads((d / "seed1.json").read_text(encoding="utf-8"))
    assert meta["input"]["mode"] == "outpaint" and meta["input"]["yaws"] == [0, 120, 240]
    assert len(meta["input"]["files"]) == 3
    assert main(args[:-2] + ["--image", imgs[0], "--image", imgs[0], "--image", imgs[0], "--image", imgs[0]]) == 1   # 4장은 거절


def test_cli_flat_image(tiny_repo, tmp_path, monkeypatch):
    """일반 이미지: 파노라마 LoRA·트리거 문구 없이 지정 비율, 업스케일은 flat_scale 배 (순환 패딩 없음)."""
    import json

    from PIL import Image

    from panogen.cli import main

    out, conf = _cli_env(tiny_repo, tmp_path, monkeypatch)
    assert main(["--config", str(conf), "generate", "-p", "misty temple", "--seed", "4", "-n", "flat",
                 "--type", "flat", "--aspect", "16:9"]) == 0
    d = out / "flat"
    meta = json.loads((d / "seed4.json").read_text(encoding="utf-8"))
    assert meta["type"] == "flat" and meta["full_prompt"] == "misty temple" and meta["lora"] == ""
    assert Image.open(d / "seed4_2k.png").size == tuple(meta["size"]) == (160, 96)
    assert "seam_verdict" not in meta
    assert main(["--config", str(conf), "upscale", str(d), "--seed", "4"]) == 0
    big = sorted(d.glob("seed4_*px.jpg"))
    assert big and Image.open(big[0]).size == (160 * 4, 96 * 4)


def test_cli_flat_with_image_auto_aspect(tiny_repo, tmp_path, monkeypatch):
    import json

    from panogen.cli import main

    out, conf = _cli_env(tiny_repo, tmp_path, monkeypatch)
    src = _photo(tmp_path / "tall.jpg", (160, 90))           # 세로 사진 → 9:16
    assert main(["--config", str(conf), "generate", "-p", "misty temple", "--seed", "2", "-n", "fi",
                 "--type", "flat", "--image", src]) == 0
    meta = json.loads((out / "fi" / "seed2.json").read_text(encoding="utf-8"))
    assert meta["input"]["mode"] == "img2img" and meta["size"][0] < meta["size"][1]


def test_cli_seam_later_then_fix(tiny_repo, tmp_path, monkeypatch):
    """360 초안: 이음새 보정 전 *_raw.png 만 → upscale --seam-only 로 *_2k.png 생성, 초안은 남김."""
    import json

    from panogen.cli import main

    out, conf = _cli_env(tiny_repo, tmp_path, monkeypatch, seam="auto")
    src = _photo(tmp_path / "p.jpg", (60, 80))
    assert main(["--config", str(conf), "generate", "-p", "misty temple", "--seed", "3", "-n", "draft",
                 "--image", src, "--seam-later"]) == 0
    d = out / "draft"
    assert (d / "seed3_raw.png").exists() and not (d / "seed3_2k.png").exists()
    meta = json.loads((d / "seed3.json").read_text(encoding="utf-8"))
    assert meta["seam_pending"] is True and "seam_verdict" in meta
    assert main(["--config", str(conf), "upscale", str(d), "--seed", "3", "--seam-only"]) == 0
    assert (d / "seed3_2k.png").exists() and (d / "seed3_raw.png").exists()
    meta = json.loads((d / "seed3.json").read_text(encoding="utf-8"))
    assert meta["seam_pending"] is False and meta["seam_inpaint"]["after"] >= 0     # 직접 고른 보정은 생성형 인페인팅
    assert not list(d.glob("seed3_*k.jpg"))                                          # --seam-only 는 8K 안 함


def test_mps_drops_transformer_before_decode(tiny_repo):
    """맥 경로: 디코딩 직전에 트랜스포머를 내리고, 다음 생성 때 다시 올려 같은 결과.
    (실제 mps 없이 분기 함수를 직접 호출해 검증)"""
    import numpy as np

    from panogen.engine import Engine
    from panogen.profiles import Profile

    base, lora = tiny_repo
    eng = Engine(ROOT, _cfg(base, lora), Profile("t", "cpu", "bf16", "cpu", False, 0, ""), log=lambda m: None)
    [emb] = eng.encode(["This is a panorama. The image shows a temple"])
    ref = np.asarray(eng.generate(emb, 3))
    pipe = eng.load()
    kw = dict(**emb, width=128, height=64, num_inference_steps=eng.sampling["steps"], **eng._call_kwargs())
    out = np.asarray(eng._drop_before_decode(pipe, {**kw, "generator": torch.Generator("cpu").manual_seed(3)}))
    assert eng.pipe is None and pipe.transformer is None
    assert np.array_equal(out, ref)
    assert np.array_equal(np.asarray(eng.generate(emb, 3)), ref)    # 다시 올려서 생성
