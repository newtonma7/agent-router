#!/usr/bin/env python3
"""Run the seed benchmark or an online policy replay without FastAPI."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from adaptive_router.application import build_service
from adaptive_router.config import Settings
from adaptive_router.experiments import ExperimentRunner, run_frozen_benchmark
from adaptive_router.experiments.report import generate_report
from adaptive_router.models import load_seed_dataset
from adaptive_router.persistence import JSONLRecorder
from adaptive_router.routing import (
    CategoryPolicy,
    EpsilonGreedyPolicy,
    LinUCBPolicy,
    RandomPolicy,
    StaticPolicy,
    UCBPolicy,
)


class TaskEvaluator:
    """Select the existing evaluator that matches each task's contract."""

    def __init__(self, evaluators: dict[str, Any]) -> None:
        self.evaluators = evaluators

    def evaluate(self, task: Any, answer: Any) -> Any:
        evaluation_type = getattr(task.evaluation_type, "value", task.evaluation_type)
        evaluator = self.evaluators[str(evaluation_type).lower()]
        return evaluator.evaluate(task, answer)


def _report_id(mode: str, run_id: str) -> str:
    now = datetime.now(timezone.utc)
    digest = hashlib.sha256(f"{mode}:{run_id}:{now.isoformat()}".encode()).hexdigest()[:10]
    return f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{digest}"


def _policy(name: str, settings: Settings, seed: int | None) -> Any:
    if name == "category":
        return CategoryPolicy()
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "epsilon":
        return EpsilonGreedyPolicy(epsilon=settings.epsilon, seed=seed)
    if name == "ucb":
        return UCBPolicy(confidence=settings.ucb_confidence)
    if name == "linucb":
        return LinUCBPolicy(alpha=settings.linucb_alpha)
    if name in {"direct", "strong", "tool"}:
        return StaticPolicy(name)
    raise ValueError(f"unknown policy: {name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("benchmark", "replay"),
        default="benchmark",
        help="benchmark every strategy or replay one routing policy (default: benchmark)",
    )
    parser.add_argument("--policy", default="linucb", help="policy for replay mode (default: linucb)")
    parser.add_argument("--seed", type=int, default=7, help="random seed for replay (default: 7)")
    parser.add_argument("--dataset", type=Path, default=Path("data/seed_tasks.json"))
    parser.add_argument(
        "--records",
        type=Path,
        help="JSONL record path (default: experiments/runs/<unique-run-id>.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="report path (default: experiments/reports/<mode>-<unique-report-id>.json)",
    )
    return parser


def _report_rows(records: list[dict[str, Any]], tasks: list[Any]) -> list[dict[str, Any]]:
    categories = {str(task.id): str(task.category.value) for task in tasks}
    rows = []
    for record in records:
        row = dict(record)
        row.setdefault("cost_usd", row.get("estimated_cost_usd"))
        row.setdefault("category", categories.get(str(row.get("task_id", "")), ""))
        rows.append(row)
    return rows


def _run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings.from_env()
    service = build_service(settings)
    dataset = load_seed_dataset(args.dataset)
    run_id = uuid.uuid4().hex
    report_id = _report_id(args.mode, run_id)
    records_path = args.records or Path("experiments/runs") / f"{args.mode}-{report_id}.jsonl"
    recorder = JSONLRecorder(records_path)
    tasks = dataset.tasks

    if args.mode == "benchmark":
        result = run_frozen_benchmark(
            args.dataset,
            service.strategies,
            evaluators=service.evaluators,
            recorder=recorder,
            run_id=run_id,
            reference_cost_usd=service.reference_cost_usd,
            reference_latency_seconds=service.reference_latency_seconds,
            cost_penalty=service.cost_penalty,
            latency_penalty=service.latency_penalty,
        )
        records = result["records"]
        metadata = {
            "mode": args.mode,
            "dataset": str(args.dataset),
            "dataset_version": result["dataset_version"],
            "benchmark_run_id": result["run_id"],
            "run_id": run_id,
            "records": str(records_path),
        }
    else:
        policy = _policy(args.policy, settings, args.seed)
        result = ExperimentRunner(
            service.strategies,
            evaluator=TaskEvaluator(service.evaluators),
            recorder=recorder,
            reference_cost_usd=service.reference_cost_usd,
            reference_latency_seconds=service.reference_latency_seconds,
            cost_penalty=service.cost_penalty,
            latency_penalty=service.latency_penalty,
        ).run(
            tasks,
            policy,
            seed=args.seed,
            run_id=run_id,
            policy_name=args.policy,
            configuration={"settings": service.configuration},
        )
        records = result.records
        metadata = {
            "mode": args.mode,
            "policy": args.policy,
            "dataset": str(args.dataset),
            "dataset_version": dataset.version,
            "seed": args.seed,
            "run_id": result.run_id,
            "records": str(records_path),
        }

    report_records = _report_rows(records, tasks)
    report = generate_report(report_records)
    report["experiment"] = {
        **metadata,
        "report_id": report_id,
        "errors": sum(1 for row in records if row.get("error")),
    }
    report["settings"] = service.configuration
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = _run(args)
    output = args.output or Path("experiments/reports") / f"{args.mode}-{report['experiment']['report_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    aggregate = report["aggregate"]
    errors = report["experiment"]["errors"]
    print(
        f"Wrote {output}: {aggregate['attempts']} attempts, "
        f"quality={aggregate['quality']:.3f}, pass_rate={aggregate['pass_rate']:.3f}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
