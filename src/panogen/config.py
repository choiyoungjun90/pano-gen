"""설정: config/default.toml → config/local.toml → --config 순서로 병합. .env 도 읽음."""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any


def find_root() -> Path:
    for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents, Path(__file__).resolve().parents[2]]:
        if (p / "config" / "default.toml").exists() and (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("저장소 루트를 찾을 수 없습니다. pano-gen 폴더 안에서 실행하세요.")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_env(root: Path) -> None:
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        v = v.strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


class Config(dict):
    root: Path

    def get_path(self, value: str) -> Path:
        p = Path(str(value).replace("\\", "/"))
        return p if p.is_absolute() else self.root / p


def load_config(extra: Path | None = None, root: Path | None = None) -> Config:
    root = root or find_root()
    load_env(root)
    with open(root / "config" / "default.toml", "rb") as f:
        data = tomllib.load(f)
    for p in (root / "config" / "local.toml", extra):
        if p and Path(p).exists():
            with open(p, "rb") as f:
                data = _merge(data, tomllib.load(f))
    cfg = Config(data)
    cfg.root = root
    return cfg
