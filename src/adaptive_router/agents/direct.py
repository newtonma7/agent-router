"""Cheap direct strategy."""

from __future__ import annotations

from typing import Any

from adaptive_router.models.agent_result import AgentStrategy
from adaptive_router.output import task_output_contract
from adaptive_router.providers.base import CompletionResponse

from .base import MeasuredStrategy, Pricing, _prompt


class DirectStrategy(MeasuredStrategy):
    """Send the task prompt once using the configured direct model."""

    strategy = AgentStrategy.DIRECT

    def _complete(self, task: Any) -> tuple[CompletionResponse, int]:
        system_prompt, response_format = task_output_contract(task)
        return self.provider.complete(
            _prompt(task),
            model=self.model,
            system_prompt=system_prompt,
            response_format=response_format,
        ), 0


DirectAgent = DirectStrategy

__all__ = ["DirectAgent", "DirectStrategy"]
