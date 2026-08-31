from adaptive_router.agents.direct import DirectStrategy
from adaptive_router.agents.tool import ToolStrategy
from adaptive_router.models import EvaluationType, Task, TaskCategory
from adaptive_router.providers.openai import OpenAICompatibleProvider


def test_direct_strategy_sends_machine_output_contract():
    requests = []

    def transport(url, headers, payload, timeout):
        requests.append(payload)
        return {
            "choices": [{"message": {"content": '{"answer":4}', "tool_calls": []}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    task = Task(
        id="A1",
        prompt="What is 2 + 2?",
        category=TaskCategory.ARITHMETIC,
        evaluation_type=EvaluationType.NUMERIC,
        expected_answer=4,
    )
    provider = OpenAICompatibleProvider(api_key="secret", transport=transport)
    result = DirectStrategy(provider, model="model").execute(task)

    assert result.error is None
    assert requests[0]["messages"][0]["role"] == "system"
    assert requests[0]["response_format"]["type"] == "json_schema"


def test_tool_strategy_does_not_offer_calculator_to_non_arithmetic_tasks():
    requests = []

    def transport(url, headers, payload, timeout):
        requests.append(payload)
        return {
            "choices": [{"message": {"content": '{"answer":"No"}', "tool_calls": []}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    task = Task(
        id="R1",
        prompt="Can Zed be a tal?",
        category=TaskCategory.REASONING,
        evaluation_type=EvaluationType.EXACT,
        expected_answer="No",
    )
    provider = OpenAICompatibleProvider(api_key="secret", transport=transport)
    result = ToolStrategy(provider, model="model").execute(task)

    assert result.error is None
    assert "tools" not in requests[0]
    assert "parallel_tool_calls" not in requests[0]


def test_tool_strategy_serializes_calculator_calls():
    requests = []

    def transport(url, headers, payload, timeout):
        requests.append(payload)
        return {
            "choices": [{"message": {"content": '{"answer":4}', "tool_calls": []}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    task = Task(
        id="A1",
        prompt="What is 2 + 2?",
        category=TaskCategory.ARITHMETIC,
        evaluation_type=EvaluationType.NUMERIC,
        expected_answer=4,
    )
    provider = OpenAICompatibleProvider(api_key="secret", transport=transport)
    result = ToolStrategy(provider, model="model").execute(task)

    assert result.error is None
    assert requests[0]["parallel_tool_calls"] is False
