"""Higher-capability strategy."""

from __future__ import annotations

from typing import Any

from adaptive_router.models.agent_result import AgentStrategy
from adaptive_router.providers.base import CompletionResponse

from .base import MeasuredStrategy, Pricing, _prompt


class StrongStrategy(MeasuredStrategy):
    """Use the same prompt/execution semantics with a stronger model setting."""

    strategy = AgentStrategy.STRONG

    def _complete(self, task: Any) -> tuple[CompletionResponse, int]:
        return self.provider.complete(_prompt(task), model=self.model), 0


StrongAgent = StrongStrategy

__all__ = ["StrongAgent", "StrongStrategy"]
