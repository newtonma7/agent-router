from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .base import ACTIONS, Policy, _check_action, update_arguments


class UCBPolicy(Policy):
    def __init__(self, exploration: float = 2.0, c: float | None = None, confidence: float | None = None, actions: Sequence[str] = ACTIONS):
        exploration = exploration if c is None else c
        exploration = exploration if confidence is None else confidence
        if exploration < 0:
            raise ValueError("exploration must be non-negative")
        self.exploration = float(exploration)
        self.c = self.exploration
        self.actions = tuple(actions)
        if not self.actions:
            raise ValueError("at least one action is required")
        self.counts = {action: 0 for action in self.actions}
        self.values = {action: 0.0 for action in self.actions}

    def select(self, context: Sequence[float] | None = None) -> str:
        for action in self.actions:
            if self.counts[action] == 0:
                return action
        total = sum(self.counts.values())
        return max(
            self.actions,
            key=lambda action: (
                self.values[action] / self.counts[action] + self.exploration * math.sqrt(math.log(total) / self.counts[action]),
                -self.actions.index(action),
            ),
        )

    def update(self, context_or_action: Any, action_or_reward: Any, reward: float | None = None) -> None:
        action, reward_value, _ = update_arguments(context_or_action, action_or_reward, reward)
        name = _check_action(action)
        if name not in self.counts:
            raise ValueError(f"action {name!r} is not configured")
        self.counts[name] += 1
        self.values[name] += (reward_value - self.values[name]) / self.counts[name]


UCB = UCBPolicy
UCBRouter = UCBPolicy
