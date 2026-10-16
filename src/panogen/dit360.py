"""외부 저장소 DiT360(MIT) 의 DiT360Pipeline 을 불러온다.

third_party/DiT360 은 scripts/fetch_dit360.py 가 고정 커밋으로 받는다 (git 에는 포함 안 됨).
DiT360 의 `src` 패키지 이름이 흔해서 sys.path 오염을 피하려고 파일 경로로 직접 로드한다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

DIT360_REPO = "https://github.com/Insta360-Research-Team/DiT360.git"
DIT360_COMMIT = "3779fe7965473f6824994c663a0ae7a76bc7aafa"  # 2026-04-17, 검증한 버전


def pipeline_path(root: Path) -> Path:
    return root / "third_party" / "DiT360" / "src" / "pipeline.py"


def load_pipeline_class(root: Path):
    path = pipeline_path(root)
    if not path.exists():
        raise FileNotFoundError(
            "DiT360 코드가 없습니다. 먼저 `python scripts/fetch_dit360.py` 를 실행하세요.")
    name = "panogen_vendor_dit360_pipeline"
    if name in sys.modules:
        return sys.modules[name].DiT360Pipeline
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.DiT360Pipeline
