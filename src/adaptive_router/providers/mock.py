"""Deterministic provider for tests and local mock mode."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import CompletionResponse, ProviderError, ToolCall


class MockProviderError(ProviderError):
    """A deliberately configured mock failure."""


class MockProvider:
    """A no-network provider with repeatable responses.

    ``responses`` can be a sequence (consumed in order), a prompt mapping, or
    a single response. A response may be text or :class:`CompletionResponse`.
    With no response configured, the provider echoes the prompt deterministically.
    """

    def __init__(
        self,
        response: str | CompletionResponse | None = None,
        *,
        responses: Sequence[str | CompletionResponse] | Mapping[str, str | CompletionResponse] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
    ) -> None:
        if response is not None and responses is not None:
            raise ValueError("provide response or responses, not both")
        self._response = response
        self._responses = list(responses) if isinstance(responses, Sequence) and not isinstance(responses, (str, bytes)) else responses
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._error = error
        self.calls: list[tuple[str, str]] = []
        self.tool_results: list[tuple[Mapping[str, Any], ...]] = []

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        tools: Sequence[Mapping[str, Any]] = (),
        tool_results: Sequence[Mapping[str, Any]] = (),
        system_prompt: str | None = None,
        response_format: Mapping[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> CompletionResponse:
        self.calls.append((prompt, model))
        self.tool_results.append(tuple(tool_results))
        if self._error is not None:
            raise MockProviderError(self._error)

        configured: str | CompletionResponse | None = self._response
        if isinstance(self._responses, list):
            if not self._responses:
                raise MockProviderError("mock response sequence exhausted")
            configured = self._responses.pop(0)
        elif isinstance(self._responses, Mapping):
            configured = self._responses.get(prompt, self._response)

        if configured is None:
            return CompletionResponse(
                text=f"mock: {prompt}",
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
            )
        if isinstance(configured, CompletionResponse):
            return configured
        return CompletionResponse(
            text=configured,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


def tool_response(
    name: str,
    arguments: Any,
    *,
    call_id: str = "mock-call",
) -> CompletionResponse:
    """Convenience for a deterministic scripted tool response."""
    return CompletionResponse(tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),))
