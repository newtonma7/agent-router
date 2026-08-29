from types import SimpleNamespace

from adaptive_router.experiments import ExperimentRunner, run_frozen_benchmark
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


def test_frozen_benchmark_runs_each_strategy_on_all_seed_tasks(tmp_path):
    from adaptive_router.models import load_seed_dataset
    from adaptive_router.persistence import JSONLRecorder

    tasks = load_seed_dataset("data/seed_tasks.json").tasks
    strategies = {name: Strategy(name, 1) for name in ("direct", "strong", "tool")}
    result = run_frozen_benchmark(
        "data/seed_tasks.json",
        strategies,
        recorder=JSONLRecorder(tmp_path / "benchmark.jsonl"),
    )

    assert len(result["records"]) == len(tasks) * len(strategies)
    assert len((tmp_path / "benchmark.jsonl").read_text().splitlines()) == 36


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


def test_replay_uses_agent_answer_for_builtin_evaluator():
    from adaptive_router.evaluation import ExactEvaluator
    from adaptive_router.models import Task

    task = Task(
        id="R1",
        prompt="Can Zed be a tal? Answer yes or no.",
        category="reasoning",
        evaluation_type="exact",
        expected_answer="direct",
    )
    strategy = Strategy("direct", 1)
    result = ExperimentRunner(
        {"direct": strategy, "strong": strategy, "tool": strategy},
        ExactEvaluator(),
    ).run([task], StaticPolicy("direct"))

    assert result.records[0]["evaluation"]["passed"] is True


def test_replay_computes_hindsight_regret_from_available_strategies():
    task = SimpleNamespace(id="a", category="arithmetic", prompt="2 + 2")
    strategies = {
        "direct": Strategy("direct", 0.4),
        "strong": Strategy("strong", 1.0),
        "tool": Strategy("tool", 0.8),
    }
    result = ExperimentRunner(strategies, Evaluator()).run(
        [task],
        StaticPolicy("direct"),
        oracle_candidates={
            "a": [
                {"quality": 0.4, "passed": False, "cost_usd": 0.01, "latency_seconds": 0.01},
                {"quality": 1.0, "passed": True, "cost_usd": 0.01, "latency_seconds": 0.01},
            ]
        },
    )

    assert result.cumulative_regret > 0
    assert result.records[0]["regret"] > 0


def test_replay_policy_randomness_is_seeded_by_run_seed():
    from adaptive_router.routing import RandomPolicy

    tasks = [SimpleNamespace(id=str(i), category="arithmetic", prompt="2 + 2") for i in range(12)]
    strategies = {name: Strategy(name, 1) for name in ("direct", "strong", "tool")}
    first = ExperimentRunner(strategies).run(tasks, RandomPolicy(seed=1), seed=42)
    second = ExperimentRunner(strategies).run(tasks, RandomPolicy(seed=99), seed=42)

    assert [row["action"] for row in first.records] == [row["action"] for row in second.records]


def test_reports_include_aggregate_task_and_category_metrics():
    from adaptive_router.experiments import aggregate_report, category_report, task_report

    rows = [
        {"task_id": "a", "category": "arithmetic", "policy": "direct", "quality": 1, "passed": True, "cost_usd": .1, "latency_seconds": .2, "reward": .8},
        {"task_id": "b", "category": "reasoning", "policy": "strong", "quality": .5, "passed": False, "cost_usd": .2, "latency_seconds": .3, "reward": .2},
    ]
    assert aggregate_report(rows)["attempts"] == 2
    assert task_report(rows)["a"]["pass_rate"] == 1
    assert category_report(rows)["reasoning"]["quality"] == .5


def test_reports_do_not_treat_unscored_attempts_as_quality_failures():
    from adaptive_router.experiments import aggregate_report

    report = aggregate_report(
        [
            {"quality": 1.0, "passed": True, "cost_usd": 0.1, "latency_seconds": 0.2, "reward": 0.8},
            {"quality": None, "passed": None, "cost_usd": None, "latency_seconds": 0.3, "reward": None, "error": "judge failed"},
        ]
    )

    assert report["attempts"] == 2
    assert report["scored_attempts"] == 1
    assert report["unscored_attempts"] == 1
    assert report["quality"] == 1.0
    assert report["pass_rate"] == 1.0
    assert report["cost_usd"] == 0.1
    assert report["errors"] == 1
