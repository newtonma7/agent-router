"""Small provider boundary shared by all strategies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class ProviderError(RuntimeError):
    """Base class for failures returned by a model provider."""


class ProviderConfigurationError(ProviderError):
    """Provider cannot be used with its current configuration."""


class ProviderRequestError(ProviderError):
    """The provider rejected or could not receive a request."""


class ProviderResponseError(ProviderError):
    """The provider returned an unusable response."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Any


@dataclass(frozen=True)
class CompletionResponse:
    """Normalized response independent of a vendor SDK."""

    text: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None


class Provider(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        model: str,
        tools: Sequence[Mapping[str, Any]] = (),
        tool_results: Sequence[Mapping[str, Any]] = (),
        system_prompt: str | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> CompletionResponse:
        """Complete one prompt without retries or fallback."""
