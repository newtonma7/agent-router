"""Reproducible aggregate, per-task, and per-category JSONL reports."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from adaptive_router.persistence import load_records


def _records(source: Iterable[Mapping[str, Any]] | str | Path | Any) -> list[Mapping[str, Any]]:
    if isinstance(source, (str, Path)):
        return load_records(source)
    if hasattr(source, "records"):
        return list(source.records)
    return list(source)


def _metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    average = lambda name: sum(float(row.get(name) or 0.0) for row in rows) / count if count else 0.0
    return {
        "attempts": count,
        "quality": average("quality"),
        "pass_rate": sum(bool(row.get("passed")) for row in rows) / count if count else 0.0,
        "cost_usd": average("cost_usd"),
        "latency_seconds": average("latency_seconds"),
        "reward": average("reward"),
        "regret": sum(float(row.get("regret") or 0.0) for row in rows),
    }


def aggregate_report(source: Iterable[Mapping[str, Any]] | str | Path | Any) -> dict[str, Any]:
    rows = _records(source)
    policies: dict[str, dict[str, Any]] = {}
    for policy in sorted({str(row.get("policy", "")) for row in rows}):
        policies[policy] = _metrics([row for row in rows if str(row.get("policy", "")) == policy])
    result = _metrics(rows)
    result["policies"] = policies
    result["limitations"] = [
        "The frozen seed benchmark is development evidence, not a production-scale claim.",
        "Live strategy stochasticity and rubric-judge calibration can affect results.",
        "Small data and distribution shift limit generalization.",
    ]
    return result


def task_report(source: Iterable[Mapping[str, Any]] | str | Path | Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _records(source):
        groups[str(row.get("task_id", ""))].append(row)
    return {key: _metrics(value) for key, value in sorted(groups.items())}


def category_report(source: Iterable[Mapping[str, Any]] | str | Path | Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _records(source):
        groups[str(row.get("category", ""))].append(row)
    return {key: _metrics(value) for key, value in sorted(groups.items())}


def generate_report(source: Iterable[Mapping[str, Any]] | str | Path | Any) -> dict[str, Any]:
    rows = _records(source)
    return {
        "aggregate": aggregate_report(rows),
        "per_task": task_report(rows),
        "per_category": category_report(rows),
    }


report_from_jsonl = generate_report
