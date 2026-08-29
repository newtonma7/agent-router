import json

from adaptive_router.persistence import JSONLRecorder, RunRecord, load_records


def test_jsonl_recorder_appends_complete_json_records(tmp_path):
    path = tmp_path / "runs.jsonl"
    recorder = JSONLRecorder(path)
    recorder.append(RunRecord(task_id="t1", policy="random", context=(1.0, 0.0), action="direct", answer="ok", run_id="r1"))
    recorder.append({"task_id": "t2", "policy": "random", "context": [], "action": "tool"})
    rows = load_records(path)
    assert [row["task_id"] for row in rows] == ["t1", "t2"]
    assert rows[0]["context"] == [1.0, 0.0]
    assert all(path.read_text().count("\n") == 2 for _ in [0])


def test_run_record_serializes_nested_model_like_values(tmp_path):
    path = tmp_path / "runs.jsonl"
    JSONLRecorder(path).append(RunRecord(task_id="t", policy="p", context=[], action="direct", evaluation={"quality": .5}))
    row = json.loads(path.read_text())
    assert row["evaluation"]["quality"] == .5
