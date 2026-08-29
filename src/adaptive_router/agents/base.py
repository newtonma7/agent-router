"""Common measured strategy execution contract."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from adaptive_router.models.agent_result import AgentResult, AgentStrategy
from adaptive_router.providers.base import CompletionResponse, Provider


class StrategyExecutionError(RuntimeError):
    """A strategy could not produce an answer."""


class Strategy(Protocol):
    def execute(self, task: Any) -> AgentResult:
        """Execute a task and return a measured result, including failures."""


@dataclass(frozen=True)
class Pricing:
    input_per_1k_tokens: float = 0.0
    output_per_1k_tokens: float = 0.0

    def estimate(self, response: CompletionResponse) -> float | None:
        if response.input_tokens is None or response.output_tokens is None:
            return None
        return (
            response.input_tokens * self.input_per_1k_tokens / 1000
            + response.output_tokens * self.output_per_1k_tokens / 1000
        )


def _task_id(task: Any) -> str:
    value = getattr(task, "id", None)
    if not isinstance(value, str) or not value:
        raise StrategyExecutionError("task must have a non-empty string id")
    return value


def _prompt(task: Any) -> str:
    value = getattr(task, "prompt", None)
    if not isinstance(value, str):
        raise StrategyExecutionError("task must have a string prompt")
    return value


class MeasuredStrategy:
    """Shared measurement/error handling for direct, strong, and tool agents."""

    strategy: AgentStrategy

    def execute(self, task: Any) -> AgentResult:
        started = time.perf_counter()
        task_id = getattr(task, "id", "")
        calls = 0
        try:
            task_id = _task_id(task)
            response, calls = self._complete(task)
            if not isinstance(response, CompletionResponse):
                raise StrategyExecutionError("provider returned an invalid completion")
            return self._make_result(task_id, response, time.perf_counter() - started, calls)
        except Exception as exc:
            return self._make_failure(
                str(task_id),
                time.perf_counter() - started,
                int(getattr(exc, "tool_calls", calls)),
                exc,
            )

    # ``run`` is a convenience alias for callers that use a runner vocabulary.
    run = execute

    def _complete(self, task: Any) -> tuple[CompletionResponse, int]:
        raise NotImplementedError

    def _make_result(
        self,
        task_id: str,
        response: CompletionResponse,
        latency: float,
        calls: int,
    ) -> AgentResult:
        if response.text is None:
            raise StrategyExecutionError("provider returned no final answer")
        return AgentResult(
            task_id=task_id,
            strategy=self.strategy,
            answer=response.text,
            latency_seconds=latency,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost_usd=self.pricing.estimate(response),
            tool_calls=calls,
            error=None,
        )

    def _make_failure(self, task_id: str, latency: float, calls: int, exc: Exception) -> AgentResult:
        error_type = type(exc).__name__
        return AgentResult(
            task_id=task_id,
            strategy=self.strategy,
            answer=None,
            latency_seconds=latency,
            input_tokens=None,
            output_tokens=None,
            estimated_cost_usd=None,
            tool_calls=calls,
            error=f"{error_type}: {exc}",
        )

    def __init__(
        self,
        provider: Provider,
        model: str | None = None,
        pricing: Pricing | None = None,
        *,
        settings: Any | None = None,
    ) -> None:
        strategy_name = self.strategy.value
        if settings is not None:
            model = model or getattr(settings, f"{strategy_name}_model", None)
            pricing = pricing or Pricing(
                getattr(settings, f"{strategy_name}_input_cost_per_1k_tokens", 0.0),
                getattr(settings, f"{strategy_name}_output_cost_per_1k_tokens", 0.0),
            )
        if not model:
            raise ValueError("model must be non-empty")
        self.provider = provider
        self.model = model
        self.pricing = pricing or Pricing()
