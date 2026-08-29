import json
import os
import subprocess
import sys
from pathlib import Path


def test_experiment_cli_writes_benchmark_report(tmp_path):
    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["ADAPTIVE_ROUTER_MOCK_MODE"] = "true"
    env["ADAPTIVE_ROUTER_PERSISTENCE_PATH"] = str(tmp_path / "runs.jsonl")
    output = tmp_path / "report.json"

    completed = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--mode", "benchmark", "--output", str(output)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["aggregate"]["attempts"] == 36
    assert report["experiment"]["mode"] == "benchmark"
    assert report["experiment"]["report_id"]
    assert report["experiment"]["errors"] == 0


def test_experiment_cli_creates_unique_default_reports(tmp_path):
    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["ADAPTIVE_ROUTER_MOCK_MODE"] = "true"
    command = [
        sys.executable,
        str(root / "scripts/run_experiment.py"),
        "--mode",
        "replay",
        "--policy",
        "linucb",
        "--dataset",
        str(root / "data/seed_tasks.json"),
    ]

    for _ in range(2):
        completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr

    reports = sorted((tmp_path / "experiments" / "reports").glob("replay-*.json"))
    assert len(reports) == 2
    assert reports[0].name != reports[1].name
    runs = sorted((tmp_path / "experiments" / "runs").glob("replay-*.jsonl"))
    assert len(runs) == 2
