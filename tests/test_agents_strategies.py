from types import SimpleNamespace

from adaptive_router.agents.direct import DirectStrategy
from adaptive_router.agents.strong import StrongStrategy
from adaptive_router.agents.tool import ToolStrategy
from adaptive_router.providers.base import CompletionResponse, ToolCall
from adaptive_router.providers.mock import MockProvider


def task(prompt="What is 2 + 2?"):
    return SimpleNamespace(id="t1", prompt=prompt)


def test_direct_and_strong_share_execution_shape_but_use_configured_models():
    provider = MockProvider(response="4", input_tokens=3, output_tokens=1)
    direct = DirectStrategy(provider, model="cheap")
    strong = StrongStrategy(provider, model="capable")

    direct_result = direct.execute(task())
    strong_result = strong.execute(task())

    assert direct_result.strategy.value == "direct"
    assert strong_result.strategy.value == "strong"
    assert direct_result.answer == strong_result.answer == "4"
    assert direct_result.error is None
    assert direct_result.input_tokens == 3
    assert direct_result.output_tokens == 1
    assert provider.calls == [("What is 2 + 2?", "cheap"), ("What is 2 + 2?", "capable")]


def test_tool_strategy_records_calls_and_returns_final_answer():
    provider = MockProvider(
        responses=[
            CompletionResponse(tool_calls=(ToolCall(id="1", name="calculator", arguments={"expression": "2 + 2"}),)),
            CompletionResponse(text="4"),
        ]
    )
    result = ToolStrategy(provider, model="tool", max_tool_calls=1).execute(task())

    assert result.answer == "4"
    assert result.tool_calls == 1
    assert result.error is None


def test_tool_strategy_turns_malformed_calls_into_typed_failure_result():
    provider = MockProvider(
        response=CompletionResponse(
            tool_calls=(ToolCall(id="1", name="calculator", arguments={"not_expression": "2 + 2"}),)
        )
    )
    result = ToolStrategy(provider, model="tool").execute(task())

    assert result.answer is None
    assert result.tool_calls == 1
    assert result.error is not None
    assert "Tool" in result.error


def test_tool_strategy_does_not_exceed_call_bound():
    provider = MockProvider(
        responses=[
            CompletionResponse(tool_calls=(ToolCall(id="1", name="calculator", arguments={"expression": "1"}),)),
            CompletionResponse(tool_calls=(ToolCall(id="2", name="calculator", arguments={"expression": "1"}),)),
        ]
    )
    result = ToolStrategy(provider, model="tool", max_tool_calls=1).execute(task())

    assert result.answer is None
    assert result.tool_calls == 1
    assert "maximum" in (result.error or "").lower()
