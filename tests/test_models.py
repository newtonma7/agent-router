import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_router.models import (
    AgentResult,
    AgentStrategy,
    EvaluationResult,
    EvaluationRubric,
    EvaluationType,
    SeedDataset,
    Task,
    TaskCategory,
    load_seed_dataset,
)


RUBRIC = EvaluationRubric(
    dimensions={
        "technical_correctness": 0.4,
        "completeness": 0.25,
        "reasoning_depth": 0.15,
        "clarity": 0.1,
        "relevance_concision": 0.1,
    }
)


def test_task_contract_accepts_numeric_and_rubric_tasks() -> None:
    numeric = Task(
        id="A1",
        prompt="What is 2 + 2?",
        category=TaskCategory.ARITHMETIC,
        evaluation_type=EvaluationType.NUMERIC,
        expected_answer=4,
    )
    explanation = Task(
        id="E1",
        prompt="Explain addition.",
        category=TaskCategory.EXPLANATION,
        evaluation_type=EvaluationType.RUBRIC,
        rubric=RUBRIC,
    )
    assert numeric.expected_answer == 4
    assert explanation.rubric is not None


@pytest.mark.parametrize(
    "values",
    [
        dict(id="A", prompt="p", category="arithmetic", evaluation_type="numeric"),
        dict(id="E", prompt="p", category="explanation", evaluation_type="rubric"),
        dict(
            id="A",
            prompt="p",
            category="arithmetic",
            evaluation_type="numeric",
            expected_answer=1,
            rubric=RUBRIC,
        ),
        dict(
            id="A",
            prompt="p",
            category="reasoning",
            evaluation_type="numeric",
            expected_answer=1,
        ),
    ],
)
def test_task_rejects_invalid_grading_combinations(values: dict) -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(values)


def test_result_contracts_validate_bounds() -> None:
    result = AgentResult(
        task_id="A1",
        strategy=AgentStrategy.DIRECT,
        answer="703",
        latency_seconds=0.01,
        input_tokens=4,
        output_tokens=1,
        estimated_cost_usd=0.001,
    )
    assert result.strategy is AgentStrategy.DIRECT
    assert EvaluationResult(
        quality=0.5, passed=False, grader_type="test"
    ).component_scores == {}
    with pytest.raises(ValidationError):
        EvaluationResult(quality=1.1, passed=False, grader_type="test")


def test_seed_artifact_is_versioned_and_validated() -> None:
    dataset = load_seed_dataset(Path("data/seed_tasks.json"))
    assert dataset.version == "1.0.0"
    assert [task.id for task in dataset.tasks] == [
        "A1", "A2", "A3", "R1", "R2", "R3", "E1", "E2", "E3", "S1", "S2", "S3"
    ]
    assert {task.category for task in dataset.tasks} == set(TaskCategory)
    assert SeedDataset.model_validate_json(Path("data/seed_tasks.json").read_text()).tasks


def test_seed_artifact_has_no_router_winner_metadata() -> None:
    raw = json.loads(Path("data/seed_tasks.json").read_text())
    assert all("predicted_winner" not in task for task in raw["tasks"])
