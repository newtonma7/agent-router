"""Reward, fixed-reference normalization, and hindsight regret helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def normalize_cost(cost_usd: float, reference_cost_usd: float) -> float:
    if reference_cost_usd <= 0:
        raise ValueError("reference cost must be positive")
    return float(cost_usd) / reference_cost_usd


def normalize_latency(latency_seconds: float, reference_latency_seconds: float) -> float:
    if reference_latency_seconds <= 0:
        raise ValueError("reference latency must be positive")
    return float(latency_seconds) / reference_latency_seconds


@dataclass(frozen=True)
class RewardConfig:
    reference_cost_usd: float = 1.0
    reference_latency_seconds: float = 1.0
    cost_penalty: float = 0.1
    latency_penalty: float = 0.1


@dataclass(frozen=True)
class Reward:
    quality: float
    normalized_cost: float
    normalized_latency: float
    reward: float

    @property
    def scalar(self) -> float:
        return self.reward


def calculate_reward(
    quality: float,
    cost_usd: float,
    latency_seconds: float,
    reference_cost_usd: float = 1.0,
    reference_latency_seconds: float = 1.0,
    cost_penalty: float = 0.1,
    latency_penalty: float = 0.1,
) -> Reward:
    quality = float(quality)
    if not 0 <= quality <= 1:
        raise ValueError("quality must be between 0 and 1")
    if cost_penalty < 0 or latency_penalty < 0:
        raise ValueError("reward penalties must be non-negative")
    normalized_cost = normalize_cost(cost_usd, reference_cost_usd)
    normalized_latency = normalize_latency(latency_seconds, reference_latency_seconds)
    return Reward(
        quality=quality,
        normalized_cost=normalized_cost,
        normalized_latency=normalized_latency,
        reward=quality - cost_penalty * normalized_cost - latency_penalty * normalized_latency,
    )


def reward_from(result: Any, evaluation: Any, **kwargs: Any) -> Reward:
    return calculate_reward(
        quality=float(_get(evaluation, "quality", 0.0)),
        cost_usd=float(_get(result, "estimated_cost_usd", _get(result, "cost_usd", 0.0))),
        latency_seconds=float(_get(result, "latency_seconds", 0.0)),
        **kwargs,
    )


def _candidate_key(candidate: Any) -> tuple[bool, float, float, float]:
    quality = float(_get(candidate, "quality", _get(_get(candidate, "evaluation", None), "quality", 0.0)))
    passed = bool(_get(candidate, "passed", _get(_get(candidate, "evaluation", None), "passed", False)))
    cost = float(_get(candidate, "normalized_cost", _get(candidate, "cost_usd", _get(candidate, "cost", 0.0))))
    latency = float(_get(candidate, "normalized_latency", _get(candidate, "latency_seconds", _get(candidate, "latency", 0.0))))
    return passed, quality, -cost, -latency


def hindsight_oracle(candidates: list[Any] | tuple[Any, ...]) -> Any:
    """Return the realized best outcome, preferring passing quality then cost."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    passing = [candidate for candidate in candidates if _candidate_key(candidate)[0]]
    pool = passing or list(candidates)
    if passing:
        return max(pool, key=_candidate_key)
    return max(pool, key=lambda candidate: (_candidate_key(candidate)[1], _candidate_key(candidate)[2], _candidate_key(candidate)[3]))


def regret(policy_reward: float, candidates: list[Any] | tuple[Any, ...] | Any) -> float:
    """Compute non-negative regret against the passing-tradeoff oracle."""
    if not isinstance(candidates, (list, tuple)):
        candidates = list(candidates)
    oracle = hindsight_oracle(candidates)
    oracle_reward = _get(oracle, "reward", None)
    if oracle_reward is None:
        oracle_reward = _get(oracle, "scalar", 0.0)
    return float(oracle_reward) - float(policy_reward)


def reward_with_config(quality: float, cost_usd: float, latency_seconds: float, config: RewardConfig) -> Reward:
    return calculate_reward(quality, cost_usd, latency_seconds, **config.__dict__)


def regret_from_rewards(oracle_reward: float, policy_reward: float) -> float:
    return float(oracle_reward) - float(policy_reward)


compute_reward = calculate_reward
normalize_cost_usd = normalize_cost
normalize_latency_seconds = normalize_latency
hindsight_oracle_reward = hindsight_oracle
compute_regret = regret_from_rewards
calculate_regret = regret_from_rewards

__all__ = [
    "RewardConfig", "Reward", "calculate_reward", "compute_reward", "reward_with_config", "reward_from", "normalize_cost",
    "normalize_latency", "hindsight_oracle", "regret", "compute_regret",
]
