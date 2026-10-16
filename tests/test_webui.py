import json
import sys
import time

from panogen.webui import Job, list_results

# 실제 `panogen generate` 출력 순서를 흉내 (로그 줄 + \r 로 갱신되는 tqdm 막대)
FAKE = r'''
import sys, time
w = sys.stdout.write
w("[1/3] 텍스트 인코더 적재 (cpu) …\n")
w("[3/3] (1/2) misty temple  seed=1\n")
w("[2/3] 트랜스포머 적재 — 프로필 mps-16g (gguf) …\n")
w("Loading pipeline components...: 100%|##| 5/5 [00:01<00:00, 3.1it/s]\n")
w("      준비 완료 (29s)\n")
for i in range(1, 4):
    w(f"\r 50%|#  | {i}/20 [01:30<25:30, 90.0s/it]")
w("\n      생성 1800s\n")
w("[3/3] (2/2) misty temple  seed=2\n")
w(f"\r  5%|   | 1/20 [01:30<28:30, 90.0s/it]")
sys.stdout.flush()
time.sleep(0.3)
'''


def wait(job, timeout=10):
    t0 = time.time()
    while job.state == "running" and time.time() - t0 < timeout:
        time.sleep(0.05)


def test_job_parses_phases_and_steps(tmp_path):
    job = Job("generate", [sys.executable, "-c", FAKE], "x", tmp_path)
    wait(job)
    d = job.to_dict()
    assert d["state"] == "done"
    assert d["phase"] == "generating" and d["image"] == [2, 2]
    assert d["step"] == [1, 20] and d["eta"] == "28:30"
    assert not any("90.0s/it" in ln for ln in d["log"])          # 진행 막대는 로그에 쌓지 않음
    assert any("생성 1800s" in ln for ln in d["log"])


def test_job_cancel(tmp_path):
    job = Job("generate", [sys.executable, "-c", "import time; time.sleep(30)"], "x", tmp_path)
    job.cancel()
    wait(job)
    assert job.state == "cancelled"


def test_list_results(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    for s in (1, 2):
        (d / f"seed{s}_2k.png").write_bytes(b"x")
        (d / f"seed{s}.json").write_text(json.dumps({"seed": s, "prompt": "p", "seam_verdict": "양호"}), "utf-8")
    (d / "seed2_8k.jpg").write_bytes(b"x")
    items = {i["seed"]: i for i in list_results(tmp_path, "run")}
    assert items[1]["img8k"] is None and items[2]["img8k"] == "/outputs/run/seed2_8k.jpg"
    assert items[1]["img2k"] == "/outputs/run/seed1_2k.png"


REFINE = r"""
import sys
w = sys.stdout.write
w("[1/3] 텍스트 인코더 적재 (cpu) — qwen360 …\n")
w("[refine] (1/2) p01_seed1_2k.png · qwen360\n")
w("[2/3] 트랜스포머 적재 — qwen360 · 프로필 mps-16g (gguf) …\n")
w("      준비 완료 (40s)\n")
w("      디테일 강화: 2048px → 4096px, 타일 1024, strength 0.3, 25 steps × 0.3\n")
w("        리파인 타일 3/18\n")
sys.stdout.flush()
"""


def test_job_refine_phase(tmp_path):
    job = Job("upscale", [sys.executable, "-c", REFINE], "x", tmp_path)
    wait(job)
    d = job.to_dict()
    assert d["phase"] == "refining" and d["tile"] == [3, 18] and d["image"] == [1, 2]


def test_job_disk_guard(tmp_path, monkeypatch):
    import shutil

    import panogen.webui as w

    monkeypatch.setattr(w, "MIN_FREE_GB", 1e9)    # 항상 부족한 것으로
    job = Job("generate", [sys.executable, "-c", "import time; time.sleep(30)"], "x", tmp_path)
    wait(job)
    assert job.state == "cancelled" and job.to_dict()["reason"] == "disk"
    assert any("디스크 여유" in ln for ln in job.lines)
