"""OpenAI-compatible HTTP chat-completions adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import (
    CompletionResponse,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
    ToolCall,
)

Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


class OpenAICompatibleProvider:
    """A deliberately thin adapter for ``POST /chat/completions`` endpoints."""

    def __init__(
        self,
        api_key: str | Any | None = None,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: Transport | None = None,
        settings: Any | None = None,
    ) -> None:
        # Accepting a Settings instance keeps wiring at this boundary while
        # avoiding a dependency from providers back into application startup.
        if settings is None and api_key is not None and not isinstance(api_key, str):
            settings, api_key = api_key, None
        if settings is not None:
            api_key = api_key or getattr(settings, "openai_api_key", None)
            base_url = base_url or getattr(settings, "openai_base_url", None)
            if timeout_seconds is None:
                timeout_seconds = getattr(settings, "request_timeout_seconds", None)
        api_key = api_key if isinstance(api_key, str) else None
        base_url = base_url or "https://api.openai.com/v1"
        timeout_seconds = 60.0 if timeout_seconds is None else timeout_seconds
        if not api_key:
            raise ProviderConfigurationError("an API key is required for the live provider")
        if timeout_seconds <= 0:
            raise ProviderConfigurationError("timeout_seconds must be positive")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport

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
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        assistant_calls: list[dict[str, Any]] = []
        normalized_results: list[Mapping[str, Any]] = []
        for result in tool_results:
            if not isinstance(result, Mapping):
                raise ProviderRequestError("tool result must be an object")
            call_id = result.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ProviderRequestError("tool result is missing tool_call_id")
            normalized_results.append(result)
            if isinstance(result.get("name"), str):
                assistant_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": result["name"],
                            "arguments": (
                                result.get("arguments", "")
                                if isinstance(result.get("arguments", ""), str)
                                else json.dumps(result.get("arguments", {}))
                            ),
                        },
                    }
                )
        if assistant_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": assistant_calls})
        for result in normalized_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": str(result.get("content", "")),
                }
            )

        payload: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = list(tools)
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        body = self._post(payload)
        return self._parse_response(body)

    def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self._transport is not None:
            try:
                response = self._transport(url, headers, payload, self.timeout_seconds)
            except ProviderRequestError:
                raise
            except Exception as exc:
                raise ProviderRequestError(f"provider request failed: {exc}") from exc
            if not isinstance(response, Mapping):
                raise ProviderResponseError("provider response must be a JSON object")
            return response

        request = Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as result:
                decoded = json.loads(result.read())
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ProviderRequestError(f"provider returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError(f"provider request failed: {exc}") from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderResponseError("provider returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ProviderResponseError("provider response must be a JSON object")
        return decoded

    @staticmethod
    def _parse_response(body: Mapping[str, Any]) -> CompletionResponse:
        if "error" in body:
            detail = body["error"]
            raise ProviderRequestError(f"provider returned an error: {detail}")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise ProviderResponseError("provider response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise ProviderResponseError("provider response choice has no message")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ProviderResponseError("provider message content must be text or null")
        raw_calls = message.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ProviderResponseError("provider tool_calls must be a list")
        calls: list[ToolCall] = []
        for raw in raw_calls:
            if not isinstance(raw, Mapping):
                raise ProviderResponseError("provider tool call must be an object")
            function = raw.get("function")
            if not isinstance(function, Mapping):
                raise ProviderResponseError("provider tool call has no function")
            call_id, name = raw.get("id"), function.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str) or not name:
                raise ProviderResponseError("provider tool call has invalid identity")
            calls.append(ToolCall(id=call_id, name=name, arguments=function.get("arguments")))

        usage = body.get("usage") or {}
        if not isinstance(usage, Mapping):
            raise ProviderResponseError("provider usage must be an object")
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if input_tokens is not None and (not isinstance(input_tokens, int) or input_tokens < 0):
            raise ProviderResponseError("provider prompt_tokens must be a non-negative integer")
        if output_tokens is not None and (not isinstance(output_tokens, int) or output_tokens < 0):
            raise ProviderResponseError("provider completion_tokens must be a non-negative integer")
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ProviderResponseError("provider finish_reason must be text or null")
        return CompletionResponse(
            text=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tuple(calls),
            finish_reason=finish_reason,
        )


OpenAICompatibleAdapter = OpenAICompatibleProvider
OpenAIProvider = OpenAICompatibleProvider

__all__ = ["OpenAICompatibleAdapter", "OpenAICompatibleProvider", "OpenAIProvider"]
