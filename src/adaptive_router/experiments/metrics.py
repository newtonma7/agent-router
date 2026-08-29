"""Public report and reward metric helpers."""

from adaptive_router.routing.reward import (
    Reward,
    RewardConfig,
    calculate_regret,
    calculate_reward,
    hindsight_oracle,
    normalize_cost,
    normalize_latency,
    regret,
)

from .report import aggregate_report, category_report, generate_report, task_report

__all__ = [
    "Reward", "RewardConfig", "calculate_reward", "calculate_regret", "normalize_cost",
    "normalize_latency", "hindsight_oracle", "regret", "aggregate_report", "task_report",
    "category_report", "generate_report",
]
