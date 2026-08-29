"""Full-information execution of the frozen benchmark."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Mapping

from adaptive_router.application import ApplicationService
from adaptive_router.models import load_seed_dataset
from adaptive_router.persistence import JSONLRecorder
from adaptive_router.routing import StaticPolicy


def run_frozen_benchmark(
    path: str | Path,
    strategies: Mapping[Any, Any],
    *,
    evaluators: Mapping[Any, Any] | None = None,
    recorder: JSONLRecorder | None = None,
    run_id: str | None = None,
    reference_cost_usd: float = 1.0,
    reference_latency_seconds: float = 1.0,
    cost_penalty: float = 0.0,
    latency_penalty: float = 0.0,
) -> dict[str, Any]:
    """Run every configured strategy on every validated seed task.

    These are full-information comparison runs. They are separate from online
    replay, where only the selected strategy may update the policy.
    """
    dataset = load_seed_dataset(path)
    run_id = run_id or str(uuid.uuid4())
    records: list[dict[str, Any]] = []
    for action in strategies:
        action_name = str(getattr(action, "value", action)).lower()
        service = ApplicationService(
            policy=StaticPolicy(action_name),
            strategies=strategies,
            evaluators=evaluators,
            recorder=recorder or JSONLRecorder("runs.jsonl"),
            policy_name=action_name,
            reference_cost_usd=reference_cost_usd,
            reference_latency_seconds=reference_latency_seconds,
            cost_penalty=cost_penalty,
            latency_penalty=latency_penalty,
        )
        for task in dataset.tasks:
            response = service.infer(task, evaluate=True)
            record = response.model_dump(mode="json")
            record["benchmark_run_id"] = run_id
            records.append(record)
    return {"run_id": run_id, "dataset_version": dataset.version, "records": records}


__all__ = ["run_frozen_benchmark"]
