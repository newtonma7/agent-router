"""Shared policy contract and action helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

ACTIONS = ("direct", "strong", "tool")


def action_name(action: Any) -> str:
    return str(getattr(action, "value", action)).lower()


def _check_action(action: Any) -> str:
    name = action_name(action)
    if name not in ACTIONS:
        raise ValueError(f"unknown routing action: {action!r}")
    return name


class Policy(ABC):
    """Minimal contextual-bandit interface."""

    actions: tuple[str, ...] = ACTIONS

    @abstractmethod
    def select(self, context: Sequence[float] | None = None) -> str:
        """Select an action using context known before execution."""

    @abstractmethod
    def update(self, context_or_action: Any, action_or_reward: Any, reward: float | None = None) -> None:
        """Update from the selected action's observed reward only.

        Both ``update(context, action, reward)`` and the compact
        ``update(action, reward)`` form are accepted.
        """

    def choose(self, context: Sequence[float] | None = None) -> str:
        return self.select(context)

    def observe(self, context_or_action: Any, action_or_reward: Any, reward: float | None = None) -> None:
        self.update(context_or_action, action_or_reward, reward)


class RoutingPolicy(Policy):
    """Compatibility name for the public policy contract."""


def update_arguments(context_or_action: Any, action_or_reward: Any, reward: float | None) -> tuple[Any, float, Any]:
    if reward is None:
        return context_or_action, float(action_or_reward), None
    return action_or_reward, float(reward), context_or_action


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = ["ACTIONS", "Policy", "RoutingPolicy", "action_name", "_check_action", "update_arguments"]
