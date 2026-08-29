from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .base import ACTIONS, Policy, _check_action, update_arguments


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small positive-definite system with Gaussian elimination."""
    n = len(vector)
    a = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(a[row][column]))
        if abs(a[pivot][column]) < 1e-12:
            raise ValueError("singular LinUCB matrix")
        a[column], a[pivot] = a[pivot], a[column]
        divisor = a[column][column]
        a[column] = [value / divisor for value in a[column]]
        for row in range(n):
            if row == column:
                continue
            factor = a[row][column]
            if factor:
                a[row] = [x - factor * y for x, y in zip(a[row], a[column])]
    return [a[row][-1] for row in range(n)]


class LinUCBPolicy(Policy):
    """Linear UCB with one independent ridge model per action."""

    def __init__(
        self,
        alpha: float = 1.0,
        n_features: int | None = None,
        feature_dim: int | None = None,
        regularization: float = 1.0,
        actions: Sequence[str] = ACTIONS,
    ):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if n_features is not None and feature_dim is not None and n_features != feature_dim:
            raise ValueError("n_features and feature_dim disagree")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.alpha = float(alpha)
        self.regularization = float(regularization)
        self.n_features = n_features if n_features is not None else feature_dim
        self.actions = tuple(actions)
        if not self.actions:
            raise ValueError("at least one action is required")
        self._matrices: dict[str, list[list[float]]] = {}
        self._vectors: dict[str, list[float]] = {}
        self._last_context: tuple[float, ...] | None = None
        self.counts = {action: 0 for action in self.actions}

    @property
    def matrices(self) -> dict[str, list[list[float]]]:
        return self._matrices

    @property
    def vectors(self) -> dict[str, list[float]]:
        return self._vectors

    def _ensure_dimension(self, context: Sequence[float]) -> tuple[float, ...]:
        x = tuple(float(value) for value in context)
        if not x:
            raise ValueError("LinUCB requires a non-empty feature vector")
        if self.n_features is None:
            self.n_features = len(x)
        n_features = self.n_features
        assert n_features is not None
        if len(x) != n_features:
            raise ValueError(f"expected {n_features} features, got {len(x)}")
        if not self._matrices:
            identity = [[self.regularization if row == column else 0.0 for column in range(n_features)] for row in range(n_features)]
            self._matrices = {action: [row[:] for row in identity] for action in self.actions}
            self._vectors = {action: [0.0] * n_features for action in self.actions}
        return x

    def select(self, context: Sequence[float] | None = None) -> str:
        if context is None:
            raise ValueError("LinUCB requires a feature vector")
        x = self._ensure_dimension(context)
        self._last_context = x
        scores = {}
        for action in self.actions:
            theta = _solve(self._matrices[action], self._vectors[action])
            confidence = math.sqrt(
                max(0.0, sum(left * right for left, right in zip(x, _solve(self._matrices[action], list(x)))))
            )
            scores[action] = sum(a * b for a, b in zip(theta, x)) + self.alpha * confidence
        return max(self.actions, key=lambda action: (scores[action], -self.actions.index(action)))

    def update(self, context_or_action: Any, action_or_reward: Any, reward: float | None = None) -> None:
        """Update the selected action using its pre-action context."""
        action, reward_value, context = update_arguments(context_or_action, action_or_reward, reward)
        name = _check_action(action)
        if name not in self.counts:
            raise ValueError(f"action {name!r} is not configured")
        if context is None:
            context = self._last_context
        if context is None:
            raise ValueError("LinUCB.update requires the selected context")
        x = self._ensure_dimension(context)
        matrix = self._matrices[name]
        n_features = self.n_features
        assert n_features is not None
        for row in range(n_features):
            for column in range(n_features):
                matrix[row][column] += x[row] * x[column]
        for i, value in enumerate(x):
            self._vectors[name][i] += reward_value * value
        self.counts[name] += 1

    def update_selected(self, context: Sequence[float], action: Any, reward: float) -> None:
        self.update(context, action, reward)


LinUCB = LinUCBPolicy
LinUCBRouter = LinUCBPolicy
