"""Cheap direct strategy."""

from __future__ import annotations

from typing import Any

from adaptive_router.models.agent_result import AgentStrategy
from adaptive_router.providers.base import CompletionResponse

from .base import MeasuredStrategy, Pricing, _prompt


class DirectStrategy(MeasuredStrategy):
    """Send the task prompt once using the configured direct model."""

    strategy = AgentStrategy.DIRECT

    def _complete(self, task: Any) -> tuple[CompletionResponse, int]:
        return self.provider.complete(_prompt(task), model=self.model), 0


DirectAgent = DirectStrategy

__all__ = ["DirectAgent", "DirectStrategy"]
