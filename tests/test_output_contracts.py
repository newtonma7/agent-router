from adaptive_router.models import EvaluationType, Task, TaskCategory
from adaptive_router.output import task_output_contract, unwrap_answer


def machine_task() -> Task:
    return Task(
        id="A1",
        prompt="What is 2 + 2?",
        category=TaskCategory.ARITHMETIC,
        evaluation_type=EvaluationType.NUMERIC,
        expected_answer=4,
    )


def test_task_contract_requires_strict_answer_json():
    system_prompt, response_format = task_output_contract(machine_task())

    assert "exactly one JSON object" in system_prompt
    assert response_format is not None
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["answer"]
    assert schema["additionalProperties"] is False


def test_unwrap_answer_accepts_the_machine_contract():
    assert unwrap_answer('{"answer":{"mean":21.0}}') == {"mean": 21.0}
    assert unwrap_answer('{"mean":21.0}') == {"mean": 21.0}
