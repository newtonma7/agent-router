from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import Policy, _check_action

DEFAULT_CATEGORY_MAPPING = {
    "arithmetic": "tool",
    "reasoning": "strong",
    "explanation": "strong",
    "extraction": "direct",
}


class StaticPolicy(Policy):
    def __init__(self, action: str = "direct"):
        self.action = _check_action(action)

    def select(self, context: Sequence[float] | None = None) -> str:
        return self.action

    def update(self, context_or_action: Any, action_or_reward: Any, reward: float | None = None) -> None:
        return None


class CategoryPolicy(Policy):
    """Route by the task category encoded in the pre-action context."""

    def __init__(self, mapping: Mapping[str, str] | None = None, default: str = "direct"):
        mapping = DEFAULT_CATEGORY_MAPPING if mapping is None else mapping
        self.mapping = {str(k.value if hasattr(k, "value") else k).lower(): _check_action(v) for k, v in mapping.items()}
        self.default = _check_action(default)

    def select(self, context: Sequence[float] | None = None) -> str:
        if context is None:
            return self.default
        values = tuple(context)
        category = next((i for i, value in enumerate(values[:4]) if value), None)
        return self.mapping.get(("arithmetic", "reasoning", "explanation", "extraction")[category], self.default) if category is not None else self.default

    def update(self, context_or_action: Any, action_or_reward: Any, reward: float | None = None) -> None:
        return None


class AlwaysDirectPolicy(StaticPolicy):
    def __init__(self) -> None:
        super().__init__("direct")


class AlwaysStrongPolicy(StaticPolicy):
    def __init__(self) -> None:
        super().__init__("strong")


class AlwaysToolPolicy(StaticPolicy):
    def __init__(self) -> None:
        super().__init__("tool")


AlwaysDirect = AlwaysDirectPolicy
AlwaysStrong = AlwaysStrongPolicy
AlwaysTool = AlwaysToolPolicy
StaticRoutingPolicy = StaticPolicy
CategoryRoutingPolicy = CategoryPolicy

__all__ = ["DEFAULT_CATEGORY_MAPPING", "StaticPolicy", "CategoryPolicy", "AlwaysDirectPolicy", "AlwaysStrongPolicy", "AlwaysToolPolicy"]
