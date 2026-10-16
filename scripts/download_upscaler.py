"""업스케일 모델 받기 — Real-ESRGAN x4plus (BSD-3-Clause, 비생성형 4배).

다른 모델(4x-UltraSharp, DAT 등)을 쓰려면 models/upscale/ 에 넣고
config/local.toml 의 [upscale] model_path 만 바꾸면 된다 (spandrel 이 형식을 자동 인식).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
DEST = Path(__file__).resolve().parents[1] / "models" / "upscale" / "RealESRGAN_x4plus.pth"


def main() -> int:
    if DEST.exists():
        print(f"이미 있음: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"다운로드: {URL}")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(DEST)
    print(f"완료: {DEST} ({DEST.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
