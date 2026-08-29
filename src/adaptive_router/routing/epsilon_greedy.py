from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from .base import ACTIONS, Policy, _check_action, update_arguments


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.1, seed: int | None = None, actions: Sequence[str] = ACTIONS):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        self.epsilon = epsilon
        self.actions = tuple(actions)
        if not self.actions:
            raise ValueError("at least one action is required")
        self.counts = {action: 0 for action in self.actions}
        self.values = {action: 0.0 for action in self.actions}
        self.rng = random.Random(seed)

    def select(self, context: Sequence[float] | None = None) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return max(self.actions, key=lambda action: (self.values[action], -self.actions.index(action)))

    def update(self, context_or_action: Any, action_or_reward: Any, reward: float | None = None) -> None:
        action, reward_value, _ = update_arguments(context_or_action, action_or_reward, reward)
        name = _check_action(action)
        if name not in self.counts:
            raise ValueError(f"action {name!r} is not configured")
        self.counts[name] += 1
        self.values[name] += (reward_value - self.values[name]) / self.counts[name]

    @property
    def total_count(self) -> int:
        return sum(self.counts.values())


EpsilonGreedy = EpsilonGreedyPolicy
EpsilonGreedyRouter = EpsilonGreedyPolicy
