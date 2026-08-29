import pytest
from pydantic import ValidationError

from adaptive_router.evaluation import (
    ExactEvaluator,
    NumericEvaluator,
    RubricEvaluator,
    StructuredEvaluator,
)
from adaptive_router.models import (
    EvaluationRubric,
    EvaluationType,
    MockRubricJudge,
    RubricJudgeRequest,
    RubricJudgeResponse,
    Task,
    TaskCategory,
)


def task(evaluation_type: EvaluationType, expected=None, rubric=None) -> Task:
    category = {
        EvaluationType.NUMERIC: TaskCategory.ARITHMETIC,
        EvaluationType.EXACT: TaskCategory.REASONING,
        EvaluationType.STRUCTURED: TaskCategory.EXTRACTION,
        EvaluationType.RUBRIC: TaskCategory.EXPLANATION,
    }[evaluation_type]
    return Task(
        id="test",
        prompt="prompt",
        category=category,
        evaluation_type=evaluation_type,
        expected_answer=expected,
        rubric=rubric,
    )


def test_numeric_evaluator_handles_rounding_and_rejects_wrong_values() -> None:
    evaluator = NumericEvaluator(tolerance=0.01)
    assert evaluator.evaluate(task(EvaluationType.NUMERIC, 1077.15), "$1,077.16").passed
    result = evaluator.evaluate(task(EvaluationType.NUMERIC, 703), "704")
    assert result.quality == 0
    assert result.passed is False


def test_numeric_evaluator_compares_multiple_values() -> None:
    result = NumericEvaluator().evaluate(
        task(EvaluationType.NUMERIC, {"mean": 21.0, "sd": 7.76}),
        "mean = 21.00; population standard deviation = 7.76",
    )
    assert result.quality == 1
    assert result.passed


def test_exact_evaluator_grades_conclusion_not_explanation() -> None:
    result = ExactEvaluator().evaluate(
        task(EvaluationType.EXACT, "No"), "No, because no mavens are tals."
    )
    assert result.passed
    assert ExactEvaluator().evaluate(
        task(EvaluationType.EXACT, "Diego"), "Answer: Diego must be first."
    ).passed
    assert not ExactEvaluator().evaluate(task(EvaluationType.EXACT, "No"), "Yes").passed


def test_exact_evaluator_ignores_yes_or_no_inside_prior_explanation() -> None:
    answer = "It cannot be yes. No, because no mavens are tals."
    assert ExactEvaluator().evaluate(task(EvaluationType.EXACT, "No"), answer).passed


def test_structured_evaluator_reports_partial_credit_and_full_pass() -> None:
    evaluator = StructuredEvaluator()
    benchmark_task = task(
        EvaluationType.STRUCTURED,
        {"name": "Maya Chen", "plan": "Enterprise", "seats": 48},
    )
    partial = evaluator.evaluate(
        benchmark_task, '{"name":"Maya Chen","plan":"Free","seats":48}'
    )
    assert partial.quality == pytest.approx(2 / 3)
    assert not partial.passed
    assert evaluator.evaluate(
        benchmark_task, '{"name":"Maya Chen","plan":"Enterprise","seats":48}'
    ).passed
    malformed = evaluator.evaluate(benchmark_task, "not json")
    assert malformed.quality == 0
    assert not malformed.passed


def test_structured_evaluator_honors_exact_keys_instruction() -> None:
    extraction_task = Task(
        id="S1",
        prompt="Extract name and plan. Return valid JSON with exactly those keys.",
        category=TaskCategory.EXTRACTION,
        evaluation_type=EvaluationType.STRUCTURED,
        expected_answer={"name": "Maya", "plan": "Enterprise"},
    )

    result = StructuredEvaluator().evaluate(
        extraction_task,
        '{"name":"Maya","plan":"Enterprise","extra":"ignored"}',
    )
    assert result.quality == 1.0
    assert result.passed is False


def test_rubric_evaluator_weights_scores_and_keeps_judge_blind() -> None:
    rubric = EvaluationRubric(
        dimensions={"technical_correctness": 0.8, "clarity": 0.2},
        pass_threshold=0.75,
    )
    judge = MockRubricJudge(
        {"technical_correctness": 4, "clarity": 2},
        feedback="good",
    )
    result = RubricEvaluator(judge).evaluate(
        task(EvaluationType.RUBRIC, rubric=rubric), "an explanation"
    )
    assert result.quality == pytest.approx(0.9)
    assert result.passed
    assert result.component_scores == {"technical_correctness": 4.0, "clarity": 2.0}

    request = RubricJudgeRequest(prompt="p", answer="a", rubric=rubric)
    assert set(request.model_dump()) == {"prompt", "answer", "rubric"}


def test_rubric_judge_response_rejects_unanchored_scores() -> None:
    with pytest.raises(ValidationError):
        RubricJudgeResponse(scores={"clarity": 5})
    with pytest.raises(ValidationError):
        RubricJudgeResponse.model_validate({"scores": {"clarity": "4"}})
