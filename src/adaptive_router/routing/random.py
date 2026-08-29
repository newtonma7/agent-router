from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from .base import ACTIONS, Policy


class RandomPolicy(Policy):
    def __init__(self, seed: int | None = None, actions: Sequence[str] = ACTIONS):
        self.actions = tuple(actions)
        if not self.actions:
            raise ValueError("at least one action is required")
        self.rng = random.Random(seed)

    def select(self, context: Sequence[float] | None = None) -> str:
        return self.rng.choice(self.actions)

    def update(self, context_or_action: Any, action_or_reward: Any, reward: float | None = None) -> None:
        return None


RandomRouter = RandomPolicy
RandomRoutingPolicy = RandomPolicy
