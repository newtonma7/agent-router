from typing import Any, Protocol

from adaptive_router.models import (
    EvaluationResult,
    EvaluationType,
    RubricJudgeRequest,
    RubricJudgeResponse,
    Task,
)


class RubricJudge(Protocol):
    def evaluate(self, request: RubricJudgeRequest) -> RubricJudgeResponse:
        """Return one anchored score from 0 through 4 per rubric dimension."""


class RubricEvaluator:
    def __init__(self, judge: RubricJudge) -> None:
        self.judge = judge

    def evaluate(self, task: Task, answer: Any) -> EvaluationResult:
        if task.evaluation_type is not EvaluationType.RUBRIC or task.rubric is None:
            raise ValueError("RubricEvaluator requires a task with a rubric")

        request = RubricJudgeRequest(
            prompt=task.prompt,
            answer=answer,
            rubric=task.rubric,
        )
        raw_response = self.judge.evaluate(request)
        response = RubricJudgeResponse.model_validate(raw_response)
        expected_dimensions = set(task.rubric.dimensions)
        received_dimensions = set(response.scores)
        if received_dimensions != expected_dimensions:
            missing = sorted(expected_dimensions - received_dimensions)
            extra = sorted(received_dimensions - expected_dimensions)
            raise ValueError(
                f"judge dimensions do not match rubric (missing={missing}, extra={extra})"
            )

        total_weight = sum(task.rubric.dimensions.values())
        quality = sum(
            (response.scores[dimension] / 4.0) * weight
            for dimension, weight in task.rubric.dimensions.items()
        ) / total_weight
        passed = quality >= task.rubric.pass_threshold
        return EvaluationResult(
            quality=quality,
            passed=passed,
            grader_type="rubric",
            component_scores={
                dimension: float(score)
                for dimension, score in response.scores.items()
            },
            feedback=response.feedback,
        )
