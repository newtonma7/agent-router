from types import SimpleNamespace

from adaptive_router.experiments import ExperimentRunner
from adaptive_router.routing import LinUCBPolicy, StaticPolicy


class Strategy:
    def __init__(self, action, quality):
        self.action = action
        self.quality = quality
        self.calls = []

    def execute(self, task):
        self.calls.append(task.id)
        return SimpleNamespace(
            task_id=task.id,
            strategy=self.action,
            answer=self.action,
            quality=self.quality,
            estimated_cost_usd=.01,
            latency_seconds=.01,
            error=None,
        )


class Evaluator:
    def evaluate(self, task, result):
        return SimpleNamespace(quality=result.quality, passed=result.quality == 1, grader_type="mock")


def test_replay_is_seeded_and_updates_selected_action_only():
    tasks = [SimpleNamespace(id=str(i), category="arithmetic", prompt=f"What is {i}?") for i in range(6)]
    strategies = {"direct": Strategy("direct", 1), "strong": Strategy("strong", .5), "tool": Strategy("tool", 0)}
    runner = ExperimentRunner(strategies, Evaluator())
    first = runner.run(tasks, LinUCBPolicy(alpha=0, n_features=8), seed=4, run_id="run-1")
    strategies2 = {name: Strategy(name, value) for name, value in (("direct", 1), ("strong", .5), ("tool", 0))}
    second = ExperimentRunner(strategies2, Evaluator()).run(tasks, LinUCBPolicy(alpha=0, n_features=8), seed=4, run_id="run-2")
    assert [row["task_id"] for row in first.records] == [row["task_id"] for row in second.records]
    assert [row["action"] for row in first.records] == [row["action"] for row in second.records]
    assert len(strategies["strong"].calls) + len(strategies["tool"].calls) + len(strategies["direct"].calls) == len(tasks)
    assert all("answer" not in row["context"] for row in first.records)


def test_replay_persists_run_identity_and_configuration():
    task = SimpleNamespace(id="a", category="reasoning", prompt="answer")
    strategy = Strategy("direct", 1)
    result = ExperimentRunner({"direct": strategy, "strong": strategy, "tool": strategy}, Evaluator()).run(
        [task], StaticPolicy("direct"), run_id="fixed", configuration={"trial": 2}
    )
    assert result.run_id == "fixed"
    assert result.records[0]["run_id"] == "fixed"
    assert result.records[0]["configuration"]["trial"] == 2


def test_reports_include_aggregate_task_and_category_metrics():
    from adaptive_router.experiments import aggregate_report, category_report, task_report

    rows = [
        {"task_id": "a", "category": "arithmetic", "policy": "direct", "quality": 1, "passed": True, "cost_usd": .1, "latency_seconds": .2, "reward": .8},
        {"task_id": "b", "category": "reasoning", "policy": "strong", "quality": .5, "passed": False, "cost_usd": .2, "latency_seconds": .3, "reward": .2},
    ]
    assert aggregate_report(rows)["attempts"] == 2
    assert task_report(rows)["a"]["pass_rate"] == 1
    assert category_report(rows)["reasoning"]["quality"] == .5
