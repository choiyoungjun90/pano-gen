#!/usr/bin/env bash
# macOS (Apple Silicon) 설치 — 저장소 루트에서:  bash scripts/setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "Python 3.11 이상 필요 (brew install python@3.12 → PYTHON=python3.12 bash scripts/setup_mac.sh)"; exit 1; }

"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/mac.txt
pip install -e .

python scripts/fetch_dit360.py
python scripts/download_upscaler.py

[ -f .env ] || cp .env.example .env
if [ ! -f config/local.toml ]; then
  printf '[generate]\nprofile = "mps-16g"\n\n[backends.dit360]\nsteps = 20\n\n[upscale]\ntile = 384\n' > config/local.toml
fi

echo
panogen doctor
echo
echo "다음: .env 에 HF_TOKEN 입력 → panogen download → panogen generate -p \"...\""
