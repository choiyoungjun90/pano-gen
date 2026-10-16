\xef\xbb\xbf# Windows + NVIDIA 설치 — 저장소 루트의 PowerShell 에서:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#   CUDA 빌드 변경:  -CudaTag cu126   (nvidia-smi 의 CUDA Version 이하로)
param([string]$CudaTag = "cu128")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"

if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv } else { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 이상이 필요합니다 (python.org)." }

& $py -m pip install --upgrade pip
& $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaTag"
& $py -m pip install -r requirements\cuda.txt
& $py -m pip install -e .

& $py scripts\fetch_dit360.py
& $py scripts\download_upscaler.py

if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Test-Path config\local.toml)) { Copy-Item config\local.example.toml config\local.toml }

& $py -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available())"
& .\.venv\Scripts\panogen.exe doctor
Write-Host ""
Write-Host "다음: .env 에 HF_TOKEN 입력 → panogen download → panogen generate -p `"...`""
Write-Host "다음부터는 .\.venv\Scripts\Activate.ps1 로 환경 활성화"
