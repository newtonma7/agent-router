import json
import re
from typing import Any, Protocol

from adaptive_router.output import rubric_judge_response_format
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


class ProviderRubricJudge:
    """Adapt a provider completion into the blind structured judge contract."""

    def __init__(self, provider: Any, model: str) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        self.provider = provider
        self.model = model

    def evaluate(self, request: RubricJudgeRequest) -> RubricJudgeResponse:
        dimensions = ", ".join(request.rubric.dimensions)
        prompt = (
            "Score the answer against the rubric. Return only JSON with a scores object "
            "containing one integer from 0 through 4 for each dimension and optional feedback.\n"
            f"Dimensions: {dimensions}\nPrompt: {request.prompt}\nAnswer: {request.answer}\n"
            f"Rubric guidance: {json.dumps(request.rubric.guidance, sort_keys=True)}"
        )
        response = self.provider.complete(
            prompt,
            model=self.model,
            system_prompt=(
                "You are a strict rubric judge. Return only the JSON object required by "
                "the response schema. Do not include Markdown or commentary."
            ),
            response_format=rubric_judge_response_format(request.rubric.dimensions),
        )
        if response.text is None:
            raise ValueError("rubric judge returned no response")
        text = response.text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("rubric judge returned invalid JSON") from exc
        return RubricJudgeResponse.model_validate(payload)


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
