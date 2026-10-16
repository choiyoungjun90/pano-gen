"""DiT360(MIT) 를 검증된 고정 커밋으로 third_party/DiT360 에 받는다. 맥/윈도우 공통.

git 이 있으면 clone + checkout, 없으면 GitHub 의 커밋 zip 을 받는다.
다른 커밋을 쓰려면:  python scripts/fetch_dit360.py --commit <hash>
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from panogen.dit360 import DIT360_COMMIT, DIT360_REPO  # noqa: E402

DEST = ROOT / "third_party" / "DiT360"


def via_git(commit: str) -> bool:
    if not shutil.which("git"):
        return False
    if not (DEST / ".git").exists():
        subprocess.run(["git", "clone", DIT360_REPO, str(DEST)], check=True)
    else:
        subprocess.run(["git", "-C", str(DEST), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(DEST), "checkout", "--quiet", commit], check=True)
    return True


def via_zip(commit: str) -> None:
    url = f"https://codeload.github.com/Insta360-Research-Team/DiT360/zip/{commit}"
    print(f"git 없음 → zip 다운로드: {url}")
    data = urllib.request.urlopen(url, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0]
        tmp = DEST.parent / "_dit360_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        z.extractall(tmp)
        shutil.rmtree(DEST, ignore_errors=True)
        shutil.move(str(tmp / top), str(DEST))
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=DIT360_COMMIT)
    a = ap.parse_args()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not via_git(a.commit):
        via_zip(a.commit)
    ok = (DEST / "src" / "pipeline.py").exists()
    print(f"DiT360 @ {a.commit[:10]} → {DEST}  ({'OK' if ok else 'pipeline.py 없음!'})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
