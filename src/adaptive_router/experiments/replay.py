"""Seeded online replay with honest selected-action-only learning."""

from __future__ import annotations

import inspect
import random
import time
import uuid
from types import SimpleNamespace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from adaptive_router.models.agent_result import AgentResult
    from adaptive_router.models.evaluation_result import EvaluationResult
    from adaptive_router.models.task import Task

from adaptive_router.features import extract_features
from adaptive_router.persistence import JSONLRecorder, RunRecord
from adaptive_router.routing.reward import calculate_reward, hindsight_oracle


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _name(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _strategy(strategies: Mapping[Any, Any], action: str) -> Any:
    for key, value in strategies.items():
        if _name(key) == action:
            return value
    raise KeyError(f"no strategy configured for action {action!r}")


def _call_strategy(strategy: Any, task: Any) -> Any:
    method = getattr(strategy, "execute", None) or getattr(strategy, "run", None)
    if method is None and callable(strategy):
        method = strategy
    if method is None:
        raise TypeError("strategy must be callable or expose execute/run")
    return method(task)


def _failure_result(task: Any, action: str, error: Exception, latency_seconds: float) -> Any:
    return SimpleNamespace(
        task_id=str(_get(task, "id", "")), strategy=action, answer=None,
        latency_seconds=max(0.0, latency_seconds), input_tokens=None, output_tokens=None,
        estimated_cost_usd=None, tool_calls=0, error=f"{type(error).__name__}: {error}", quality=0.0,
    )


def _call_evaluator(evaluator: Any, task: Any, result: Any) -> Any:
    if evaluator is None:
        return None
    method = getattr(evaluator, "evaluate", evaluator)
    try:
        parameters = list(inspect.signature(method).parameters.values())
        if len(parameters) >= 2 and parameters[1].name in {"result", "agent_result", "response"}:
            return method(task, result)
    except (TypeError, ValueError):
        pass
    return method(task, _record_value(result, "answer", result))


def _record_value(result: Any, name: str, default: Any = None) -> Any:
    return _get(result, name, default)


@dataclass
class ExperimentResult:
    run_id: str
    policy: str
    records: list[dict[str, Any]] = field(default_factory=list)
    cumulative_reward: float = 0.0
    cumulative_regret: float = 0.0
    seed: int | None = None
    configuration: dict[str, Any] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return self.cumulative_reward

    @property
    def regret(self) -> float:
        return self.cumulative_regret


class ExperimentRunner:
    def __init__(
        self,
        strategies: Mapping[Any, Any],
        evaluator: Any = None,
        recorder: JSONLRecorder | None = None,
        feature_extractor: Callable[[Any], Sequence[float]] = extract_features,
        reference_cost_usd: float = 1.0,
        reference_latency_seconds: float = 1.0,
        cost_penalty: float = 0.1,
        latency_penalty: float = 0.1,
    ):
        self.strategies = strategies
        self.evaluator = evaluator
        self.recorder = recorder
        self.feature_extractor = feature_extractor
        self.reward_config = {
            "reference_cost_usd": reference_cost_usd,
            "reference_latency_seconds": reference_latency_seconds,
            "cost_penalty": cost_penalty,
            "latency_penalty": latency_penalty,
        }

    def run(
        self,
        tasks: Sequence[Any],
        policy: Any,
        *,
        seed: int | None = None,
        run_id: str | None = None,
        configuration: Mapping[str, Any] | None = None,
        oracle_candidates: Mapping[str, Sequence[Any]] | None = None,
        policy_name: str | None = None,
    ) -> ExperimentResult:
        ordered_tasks = list(tasks)
        if seed is not None:
            random.Random(seed).shuffle(ordered_tasks)
            policy_rng = getattr(policy, "rng", None)
            if policy_rng is not None and hasattr(policy_rng, "seed"):
                policy_rng.seed(seed)
        run_id = run_id or str(uuid.uuid4())
        config = {**self.reward_config, **dict(configuration or {}), "seed": seed}
        records: list[dict[str, Any]] = []
        cumulative_reward = cumulative_regret = 0.0
        policy_name = policy_name or getattr(policy, "name", policy.__class__.__name__)

        for task in ordered_tasks:
            context = tuple(float(value) for value in self.feature_extractor(task))
            action = _name(policy.select(context))
            started = time.perf_counter()
            try:
                result = _call_strategy(_strategy(self.strategies, action), task)
            except Exception as error:
                result = _failure_result(task, action, error, time.perf_counter() - started)
            evaluation = _call_evaluator(self.evaluator, task, result)
            cost = _record_value(result, "estimated_cost_usd", 0.0) or 0.0
            reward = (
                calculate_reward(
                    float(_record_value(evaluation, "quality", _record_value(result, "quality", 0.0))),
                    cost,
                    _record_value(result, "latency_seconds", 0.0) or 0.0,
                    **self.reward_config,
                )
                if evaluation is None
                else calculate_reward(
                    float(_get(evaluation, "quality", 0.0)),
                    cost,
                    _record_value(result, "latency_seconds", 0.0) or 0.0,
                    **self.reward_config,
                )
            )
            policy.update(context, action, reward.reward)
            cumulative_reward += reward.reward

            candidates = (oracle_candidates or {}).get(str(_get(task, "id", "")))
            task_regret = 0.0
            if candidates:
                oracle = hindsight_oracle(list(candidates))
                oracle_reward = _get(oracle, "reward")
                if oracle_reward is None:
                    oracle_reward = calculate_reward(
                        float(_get(oracle, "quality", 0.0)),
                        float(_get(oracle, "cost_usd", 0.0)),
                        float(_get(oracle, "latency_seconds", 0.0)),
                        **self.reward_config,
                    ).reward
                task_regret = float(oracle_reward) - reward.reward
                cumulative_regret += task_regret
            record = RunRecord(
                task_id=str(_get(task, "id", "")),
                policy=str(policy_name),
                context=context,
                action=action,
                strategy=action,
                category=_name(_get(task, "category", "")),
                answer=_record_value(result, "answer"),
                evaluation=evaluation,
                quality=reward.quality,
                passed=_get(evaluation, "passed") if evaluation is not None else None,
                cost_usd=_record_value(result, "estimated_cost_usd", _record_value(result, "cost_usd", 0.0)),
                latency_seconds=_record_value(result, "latency_seconds", 0.0),
                normalized_cost=reward.normalized_cost,
                normalized_latency=reward.normalized_latency,
                reward=reward.reward,
                error=_record_value(result, "error"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                run_id=run_id,
                configuration=config,
                config=config,
                result=result,
                input_tokens=_record_value(result, "input_tokens"),
                output_tokens=_record_value(result, "output_tokens"),
                tool_calls=_record_value(result, "tool_calls"),
                regret=task_regret,
            ).to_dict()
            records.append(record)
            if self.recorder is not None:
                self.recorder.append(record)

        return ExperimentResult(run_id, str(policy_name), records, cumulative_reward, cumulative_regret, seed, config)

    replay = run


def run_policy(tasks: Sequence[Any], policy: Any, strategies: Mapping[Any, Any], **kwargs: Any) -> ExperimentResult:
    return ExperimentRunner(strategies, **{key: kwargs.pop(key) for key in list(kwargs) if key in {
        "evaluator", "recorder", "feature_extractor", "reference_cost_usd", "reference_latency_seconds", "cost_penalty", "latency_penalty"
    }}).run(tasks, policy, **kwargs)


def run_policies(tasks: Sequence[Any], policies: Mapping[str, Any], strategies: Mapping[Any, Any], **kwargs: Any) -> dict[str, ExperimentResult]:
    return {name: run_policy(tasks, policy, strategies, policy_name=name, **kwargs) for name, policy in policies.items()}
