# pano-gen

외부 API 없이 **로컬 GPU만으로** 360° 등장방형(equirectangular) 파노라마를 생성하고 **8K(8192×4096)** 로 만드는 프로젝트입니다.

- 생성: 파노라마 전용 LoRA 를 얹은 모델 3종 중 선택 (아래 "모델 백엔드")
- 디테일: 2K → 4K 를 같은 모델로 타일 img2img 해 질감을 만드는 **생성형 디테일 강화**(선택) → Real-ESRGAN x4 로 8K
- 이음새: 좌우 순환 패딩 타일 업스케일 + (Qwen) 경계를 가운데로 옮겨 같은 모델로 다시 그리는 **이음새 인페인팅**
- 맥(Apple Silicon)과 윈도우(NVIDIA)에서 같은 명령으로 동작, GPU 메모리에 맞춰 적재 방식을 자동 선택

```
프롬프트 ─▶ [1] 텍스트 인코딩 → 해제
          ─▶ [2] 트랜스포머(프로필별 bf16/fp8/nf4/gguf) + 파노라마 LoRA
          ─▶ [3] 2048×1024 생성 ─▶ 이음새 검사·(인페인팅|blend)        ← 여기까지 2K, 확인 후
          ─▶ (선택) 디테일 강화 2K→4K ─▶ Real-ESRGAN x4 (순환 패딩 타일) ─▶ 8192×4096
```

## 모델 백엔드

`config/default.toml` 의 `[backends.*]`, 기본값은 `[model] backend`, 1회만 바꾸려면 `--backend`.

| 백엔드 | 베이스 + LoRA | 맥 16GB | 특징 |
|---|---|---|---|
| `dit360` | FLUX.1-dev 12B + [DiT360](https://github.com/Insta360-Research-Team/DiT360) | 가능 (20 step 약 32분) | 추론 중 좌우 순환 패딩, 기준선 |
| `qwen360` | Qwen-Image-2512 20B + [Qwen 360 Diffusion v2](https://huggingface.co/ProGamerGov/qwen-360-diffusion) | 실험적 | 2048×1024 ERP 3.5만 장 학습. `--fast` = Lightning 8 step |
| `klein360` | FLUX.2 [klein] base 9B + [360 panorama LoRA](https://huggingface.co/crafiq/flux-2-klein-9b-360-panorama-lora) | 예상 가능 | 최신 세대·작음. HF 에서 별도 동의 필요 |

```bash
panogen download --backend qwen360 --fast           # 백엔드별로 필요한 파일만 (맥은 GGUF 트랜스포머)
panogen generate -p "..." --backend qwen360 --fast
panogen bench --backends dit360,qwen360:fast --seeds 1 2   # bench/prompts.txt 8개로 비교 → 웹 UI 에서 360° 비교
```
모델마다 LoRA 트리거 문구(`prompt_prefix`)가 자동으로 붙습니다. FLUX.1-dev·FLUX.2 klein 9B 는 비상업 라이선스입니다.

## 왜 2K 생성 → 8K 업스케일인가

DiT360 은 2048×1024 로만 학습되어 다른 크기로 생성하면 결과가 깨질 수 있다고 공식 문서에 명시되어 있습니다.
현재 공개된 어떤 파노라마 모델도 8K 를 한 번에 생성하지 못하므로, 학습 해상도에서 생성한 뒤 4배 확대하는 것이
품질과 안정성 면에서 가장 현실적인 구성입니다.

## 하드웨어 프로필

공식 DiT360 추론은 최적화 없이 약 37GB GPU 메모리를 씁니다. pano-gen 은 텍스트 인코더와 트랜스포머를
동시에 올리지 않는 staged 방식에 양자화를 더해 메모리를 줄입니다. `profile = "auto"` 면 자동 선택됩니다.

| 프로필 | 대상 | 트랜스포머 | 권장 RAM | 비고 |
|---|---|---|---|---|
| `cuda-full` | VRAM 32GB 이상 | bf16 | 32GB | 원본 품질, 가장 빠름 |
| `cuda-24g` | RTX 3090/4090 (24GB) | LoRA 병합 후 fp8 저장, bf16 연산 | 32GB | 품질 손실 매우 적음 |
| `cuda-12g` | VRAM 12~16GB | 4bit NF4 (bitsandbytes) | 32GB | 텍스트 인코딩은 CPU |
| `mps-16g` | Apple Silicon 16GB | 4bit GGUF | 16GB | **실험적**, 매우 느림 |

맥 16GB 는 돌아가더라도 장당 수십 분이 걸릴 수 있고 메모리 압박이 큽니다. 프롬프트·구도 테스트용으로 보고,
본 생성은 NVIDIA 윈도우 PC 에서 하는 것을 권장합니다.

## 설치

Python 3.11 이상, git 필요. 모델 캐시로 수십 GB 디스크가 필요합니다 (`.env` 의 `HF_HOME` 으로 위치 변경 가능).

**1) Hugging Face 준비 (한 번만)**
1. https://huggingface.co/black-forest-labs/FLUX.1-dev 에서 라이선스 동의
2. https://huggingface.co/settings/tokens 에서 Read 토큰 발급 → `.env` 의 `HF_TOKEN=` 에 입력

**2) 설치 스크립트**
```bash
# macOS
bash scripts/setup_mac.sh && source .venv/bin/activate
```
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```
스크립트가 의존성 설치, DiT360 코드 받기(고정 커밋), 업스케일 모델 받기까지 처리합니다.

**3) 점검과 모델 다운로드**
```bash
panogen doctor      # 장치·VRAM·RAM·추천 프로필·준비 상태
panogen download    # FLUX.1-dev + DiT360 LoRA (+ 맥은 GGUF) 미리 받기
```

## 사용

### 웹 UI

```bash
panogen ui            # http://127.0.0.1:8370/  (--port 로 변경)
```
| 탭 | 하는 일 |
|---|---|
| **생성** | 종류(360° 파노라마 / 일반 이미지)·프롬프트·첨부(최대 3장)·모델 선택 → 진행도 → **초안 확인**(이음새 보정 전, 평면 + 360° 미리보기) → 고른 시드만 **이음새 보정** → 8K (디테일 강화 선택), 또는 프롬프트 수정 / 새 초안. 일반 이미지는 확인 → 고해상도(4배) |
| **모델 비교** | 준비된 모델(·fast)을 골라 고정 프롬프트 N개 × 시드로 벤치 실행 → 프롬프트×모델 표로 2K·시간·메모리·이음새 비교, 칸을 누르면 360° |
| **모델 관리** | 모델별 받은 것/없는 것, 디스크 여유, 받기 버튼(Lightning 포함 선택), HF 동의 필요 여부 |

작업(생성·8K·비교·받기)은 한 번에 하나만 돌고, 매번 별도 프로세스라 끝나면 모델 메모리가 반환됩니다.
페이지를 닫아도 작업은 계속되며 다시 열면 진행 화면으로 이어집니다. `#cmp`, `#models` 로 탭을 바로 열 수 있습니다.
360° 미리보기는 three.js(jsDelivr CDN)를 불러오므로 인터넷이 필요합니다.

### 명령줄

기본 흐름은 **2K 생성 → 눈으로 확인 → 고른 시드만 8K** 입니다. 생성이 끝나면 다음에 칠 명령이 출력됩니다.

```bash
# 1) 시드 고정, 같은 프롬프트로 4장 → outputs/castle/seed100~103_2k.png + seedN.json
panogen generate -p "a medieval castle on a hilltop surrounded by autumn forests" --seed 100 -c 4 -n castle

# 2) 2K 를 확인하고 마음에 드는 시드만 8K 로 → seedN_8k.jpg (+ .png), seedN.json 에 결과 기록
panogen upscale outputs/castle --seed 101 103

# 프롬프트 파일 일괄 생성 (모델을 한 번만 올려서 연속 생성) → p01_seedN_2k.png …
panogen generate --prompts prompts.example.txt -c 2 -n batch01
panogen upscale outputs/batch01 --seed 1234        # 해당 시드의 모든 프롬프트 결과

# 확인 없이 생성 직후 8K 까지 (예전 방식). 항상 이렇게 하려면 config 에 [generate] upscale = true
panogen generate -p "..." --upscale

# 이미 있는 2:1 파노라마를 8K 로 (여러 장 가능)
panogen upscale outputs/castle/seed100_2k.png other.png
```

- 프롬프트 앞에 `This is a panorama. The image shows ` 가 자동으로 붙습니다 (DiT360 학습 캡션 형식)
- FLUX 의 T5 인코더는 영어를 가장 잘 이해하므로 영어 프롬프트를 권장합니다
- 같은 시드·설정이면 같은 결과가 나옵니다. 설정과 소요 시간은 `seedN.json` 에 기록됩니다

### 360° 초안 → 이음새 보정
웹 UI 의 360° 생성은 이음새를 보정하기 전 **초안**(`seedN_raw.png`)에서 멈춥니다. 구도가 마음에 들지 않으면 이 단계에서 프롬프트를 고치고,
고른 시드만 이음새 보정(`seedN_2k.png`, 초안은 비교용으로 남음) → 8K 로 갑니다. 8K 를 바로 고르면 이음새 보정부터 합니다.
```bash
panogen generate -p "..." --seam-later -c 3 -n draft          # 초안만 (seedN_raw.png)
panogen upscale outputs/draft --seed 2 --seam-only            # 이음새만 보정 (직접 고른 것은 생성형 인페인팅)
panogen upscale outputs/draft --seed 2                        # 이음새 보정(아직이면) → 8K
```

### 일반 이미지
`--type flat` 은 파노라마 LoRA·트리거 문구 없이 같은 베이스 모델로 일반 이미지를 만듭니다. 이음새 단계가 없고, 업스케일은 `upscale.flat_scale`(기본 4)배입니다.
```bash
panogen generate -p "a stone lantern in the rain" --type flat --aspect 16:9   # 1:1 4:3 3:4 3:2 2:3 16:9 9:16, auto = 첨부 비율
panogen generate -p "same scene at night" --type flat --image photo.jpg       # 첨부를 프롬프트대로 수정
```
크기는 약 1MP(`generate.flat_base` = 1:1 한 변, 기본 1024)입니다.

### 이미지 + 프롬프트 (최대 3장)
첨부 이미지를 프롬프트대로 고칩니다. 웹 UI 에서는 프롬프트 칸 아래 "이미지 첨부"(끌어다 놓기, 여러 장 선택 가능). 명령줄은 `--image` 를 최대 3번.
```bash
panogen generate -p "same place in heavy snow at night" --image pano.jpg --strength 0.6       # 2:1 파노라마 → 수정
panogen generate -p "a quiet korean temple courtyard" --image photo.jpg --fov 75 --strength 0.3   # 일반 사진 → 360° 확장
```
| 입력 | 처리 | `--strength` (원본 변경 정도) |
|---|---|---|
| 2:1 파노라마 | img2img | 기본 0.6. 낮을수록 원본 구도 유지, 높을수록 프롬프트 반영 |
| 일반 사진 | 사진을 정면에 투영(`--fov` = 사진 가로 화각) → 사진 밖을 새로 생성 → 사진은 원본으로 다시 합성 → strength 만큼 전체 img2img | 기본 0.3. 0 이면 사진을 그대로 둔 채 주변만 생성 |

- 폰 기본 카메라 화각은 약 70~80°, 광각은 100~120° 입니다. 화각이 실제와 다르면 정면이 늘어나거나 줄어 보입니다
- 원본은 결과 폴더에 `input1.*`, `input2.*` … 로 보관되고 `seedN.json` 의 `input` 에 방식이 기록됩니다

여러 장을 첨부했을 때:

| 모델 | 360° 파노라마 | 일반 이미지 |
|---|---|---|
| `klein360` | 전부 참조 이미지로 넣어 "Convert to an Equirectangular 360 panorama" 변환 | 전부 참조 이미지로 편집 |
| `qwen360`·`dit360` | 사진들을 방향을 나눠 배치(2장 = 0°·180°, 3장 = 0°·120°·240°, 화각은 겹치지 않게 360/장수 이하) → 나머지 생성. 2:1 파노라마가 섞여 있으면 그 1장만 수정 | 첫 장을 목표 비율로 잘라 수정, 나머지는 쓰지 않음 |

### 이음새
`qwen360` 은 추론 중 순환 패딩이 없어서, 이음새 비율이 기준(`seam.warn_ratio`) 이상이면 경계를 가운데로 옮겨
가운데 띠만 같은 모델로 다시 그립니다 (`[seam] inpaint = "auto" | "always" | "off"`, LoRA 작성자 방식).

생성 직후 좌우 경계를 검사해 톤 단차가 있으면 자동 보정합니다 (`seedN.json` 의 `seam_ratio`).
형태가 어긋난 경우는 수동으로:
```bash
panogen seam outputs/x/seed1_2k.png --roll       # 경계를 가운데로 옮긴 이미지 + 마스크
# → *_rolled.png 의 가운데 띠를 인페인팅 (DiT360 의 editing.py, ComfyUI 등)
panogen seam outputs/x/seed1_2k_rolled_inpainted.png --unroll
panogen upscale outputs/x/seed1_2k_fixed.png
```

### 디테일 강화 (생성형)
Real-ESRGAN 은 있는 픽셀을 선명하게 할 뿐 디테일을 지어내지 않아 8K 에서 뭉개져 보입니다.
`--refine` 은 2K 를 4K 로 키운 뒤 1024² 타일마다 같은 모델로 낮은 강도(기본 0.3) img2img 를 해 질감을 만들고, 그다음 8K 로 올립니다.
```bash
panogen upscale outputs/castle --seed 101 --refine              # 웹 UI 에서는 "디테일 강화" 체크
panogen upscale outputs/castle --seed 101 --refine --strength 0.25
```
- 좌우는 순환 패딩으로 처리해 이음새가 유지되고, 극점(위도 ±60° 밖)은 늘어난 영역이라 원본으로 서서히 섞습니다
- 파노라마 LoRA 는 이 단계에서 끕니다 (타일은 일반 원근 이미지 조각)
- `dit360`·`qwen360` 지원. 타일 수 × (steps × strength) 만큼 계산하므로 **맥에서는 매우 느립니다** — 3080 등 NVIDIA 권장

### 업스케일 모델 바꾸기

`models/upscale/` 에 `.pth`/`.safetensors` 를 넣고 `config/local.toml` 에서 지정합니다. spandrel 이
ESRGAN·SwinIR·DAT·HAT 등 형식을 자동 인식합니다. 모델이 없으면 Lanczos(비AI)로 확대합니다.
```toml
[upscale]
model_path = "models/upscale/4x-UltraSharp.pth"
tile = 768        # VRAM 여유가 있으면 키울수록 빠름
```

## 설정

`config/default.toml` (공용) → `config/local.toml` (머신별, git 제외) → `--config 파일` 순서로 덮어씁니다.
```toml
# config/local.toml — 윈도우 RTX 4090 예
[generate]
profile = "cuda-24g"
steps = 28
```

## 폴더 구조

```
pano-gen/
├── config/default.toml          공용 설정
├── config/local.example.toml    머신별 설정 예시
├── scripts/
│   ├── setup_mac.sh / setup_windows.ps1
│   ├── fetch_dit360.py          DiT360 을 고정 커밋으로 third_party/ 에 받기
│   └── download_upscaler.py     Real-ESRGAN x4plus 받기
├── src/panogen/
│   ├── cli.py                   명령줄
│   ├── engine.py                staged 생성 (인코딩 → 트랜스포머+LoRA → 생성), 이음새 인페인팅
│   ├── backends.py              모델 백엔드 (dit360 · qwen360 · klein360)
│   ├── refine.py                생성형 디테일 강화 (타일 img2img)
│   ├── profiles.py              하드웨어 프로필
│   ├── dit360.py                DiT360Pipeline 로더 (커밋 고정)
│   ├── seam.py                  이음새 검사·보정
│   ├── upscale.py               순환 패딩 타일 업스케일
│   └── webui.py                 웹 UI 서버 (panogen ui)
├── ui/index.html                웹 UI 페이지
├── tests/                       GPU 없이 실행 (작은 무작위 모델로 생성 경로 전체 검증)
├── bench/prompts.txt            백엔드 비교용 고정 프롬프트
├── prompts.example.txt
├── third_party/                 DiT360 (git 제외, 스크립트로 받음)
├── models/                      업스케일 가중치 (git 제외)
└── outputs/                     결과물 (git 제외)
```

## Git

- 커밋되지 않는 것: `outputs/`, `models/` 가중치, `third_party/DiT360`, `.env`, `config/local.toml`
- DiT360 은 `src/panogen/dit360.py` 의 `DIT360_COMMIT` 으로 버전이 고정되어 있어, 어느 PC 에서 받아도 같은 코드가 됩니다.
  submodule 로 관리하고 싶다면 `git submodule add https://github.com/Insta360-Research-Team/DiT360.git third_party/DiT360`
  후 해당 커밋으로 checkout 하고 `.gitignore` 의 `third_party/*` 줄을 지우면 됩니다
- `.gitattributes` 로 줄바꿈을 LF 로 고정, `.ps1` 만 CRLF + UTF-8 BOM (Windows PowerShell 5.1 한글 깨짐 방지)

```bash
git init && git add . && git commit -m "pano-gen: 로컬 360 파노라마 생성"
git branch -M main && git remote add origin <URL> && git push -u origin main
```
윈도우에서는 clone → `scripts\setup_windows.ps1` → `.env` 에 토큰 → `panogen download` 순서입니다.

## 테스트

```bash
pytest
```
작은 무작위 FLUX 구성요소로 실제 DiT360 파이프라인 코드를 CPU 에서 돌려 다음을 검사합니다:
staged 인코딩/생성, LoRA 적용 여부와 효과, fp8 병합 경로, 시드 재현성, 프롬프트 파일 일괄 생성과 파일명,
이음새 보정, 순환 패딩 업스케일. nf4(CUDA)·GGUF(맥) 적재는 해당 장비에서 첫 실행으로 확인하세요.

## 라이선스 주의

- **FLUX.1 [dev] 는 비상업 라이선스**입니다. DiT360 LoRA 도 그 위에서 동작하므로 생성물을 상업 서비스에
  쓰려면 Black Forest Labs 라이선스 조건을 사내에서 검토해야 합니다
- DiT360 코드: MIT. Real-ESRGAN: BSD-3-Clause
- 이 저장소는 모델 가중치를 포함하지 않으며, 각 사용자가 직접 동의 후 내려받습니다

## 문제 해결

| 증상 | 해결 |
|---|---|
| `401` / `gated repo` | FLUX.1-dev 동의 여부와 `.env` 의 `HF_TOKEN` 확인 |
| CUDA out of memory | 한 단계 낮은 프로필 (`--profile cuda-12g`), 다른 GPU 프로그램 종료 |
| `DiT360 LoRA 적용 실패` | `pip install -U diffusers peft` 후 재시도, 계속되면 다른 프로필 |
| 윈도우 스크립트 실행 차단 | `-ExecutionPolicy Bypass` 로 실행 |
| `bitsandbytes` 오류 (윈도우) | `pip install -U bitsandbytes` (0.45 이상은 윈도우 공식 지원) |
| 맥에서 멈춤·스왑 | 다른 앱 종료, `steps = 20`, `[upscale] tile = 256` |
| 결과가 2:1 파노라마처럼 안 보임 | 해상도를 2048×1024 로 유지했는지 확인 (DiT360 학습 해상도) |
