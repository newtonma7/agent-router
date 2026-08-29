from adaptive_router.providers.base import CompletionResponse, ProviderResponseError, ToolCall
from adaptive_router.providers.mock import MockProvider
from adaptive_router.providers.openai import OpenAICompatibleProvider


def test_mock_provider_is_repeatable_and_does_not_need_credentials():
    provider = MockProvider(response="hello", input_tokens=2, output_tokens=1)

    first = provider.complete("prompt", model="mock")
    second = provider.complete("prompt", model="mock")

    assert first == second == CompletionResponse(text="hello", input_tokens=2, output_tokens=1)


def test_openai_compatible_adapter_sends_system_prompt_and_response_format():
    requests = []

    def transport(url, headers, payload, timeout):
        requests.append(payload)
        return {
            "choices": [{"message": {"content": "answer", "tool_calls": []}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    response_format = {"type": "json_schema", "json_schema": {"name": "answer"}}
    provider = OpenAICompatibleProvider(api_key="secret", transport=transport)
    provider.complete(
        "prompt",
        model="model",
        system_prompt="system",
        response_format=response_format,
    )

    assert requests[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert requests[0]["response_format"] == response_format


def test_openai_compatible_adapter_maps_chat_completion_and_usage():
    requests = []

    def transport(url, headers, payload, timeout):
        requests.append((url, headers, payload, timeout))
        return {
            "choices": [{"message": {"content": "answer", "tool_calls": []}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }

    provider = OpenAICompatibleProvider(
        api_key="secret", base_url="https://example.test/v1", transport=transport
    )
    response = provider.complete("prompt", model="model")

    assert response == CompletionResponse(text="answer", input_tokens=4, output_tokens=2)
    assert requests[0][0] == "https://example.test/v1/chat/completions"
    assert requests[0][1]["Authorization"] == "Bearer secret"
    assert requests[0][2]["messages"] == [{"role": "user", "content": "prompt"}]


def test_openai_adapter_rejects_malformed_success_payload():
    provider = OpenAICompatibleProvider(
        api_key="secret", transport=lambda *args: {"choices": []}
    )

    try:
        provider.complete("prompt", model="model")
    except ProviderResponseError:
        pass
    else:
        raise AssertionError("malformed provider output must be a typed failure")


def test_openai_adapter_preserves_tool_calls():
    provider = OpenAICompatibleProvider(
        api_key="secret",
        transport=lambda *args: {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression":"2 + 2"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )

    assert provider.complete("prompt", model="model").tool_calls == (
        ToolCall(id="call-1", name="calculator", arguments='{"expression":"2 + 2"}'),
    )
