from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class EvaluationRubric(BaseModel):
    """Task-specific rubric weights and the quality required to pass."""

    model_config = ConfigDict(extra="forbid")

    dimensions: dict[str, float] = Field(min_length=1)
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    guidance: dict[str, str] = Field(default_factory=dict)

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, dimensions: dict[str, float]) -> dict[str, float]:
        if any(not name.strip() for name in dimensions):
            raise ValueError("rubric dimension names must not be blank")
        if any(weight <= 0 or weight > 1 for weight in dimensions.values()):
            raise ValueError("rubric weights must be greater than 0 and at most 1")
        if sum(dimensions.values()) <= 0:
            raise ValueError("rubric weights must have a positive total")
        return dimensions


class RubricJudgeRequest(BaseModel):
    """Blind input made available to a rubric judge.

    Deliberately contains no strategy, model, cost, latency, or tool metadata.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    answer: Any
    rubric: EvaluationRubric


class RubricJudgeResponse(BaseModel):
    """Structured, anchored output required from a rubric judge."""

    model_config = ConfigDict(extra="forbid")

    scores: dict[str, StrictInt] = Field(min_length=1)
    feedback: str | None = None

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, scores: dict[str, StrictInt]) -> dict[str, StrictInt]:
        if any(score < 0 or score > 4 for score in scores.values()):
            raise ValueError("rubric scores must be integers from 0 through 4")
        return scores


# A small, deterministic contract useful for tests and local runs.
class MockRubricJudge:
    def __init__(
        self,
        scores: dict[str, int] | None = None,
        *,
        default_score: int = 4,
        feedback: str | None = None,
    ) -> None:
        self.scores = scores or {}
        self.default_score = default_score
        self.feedback = feedback
        # Validate configuration at construction, not after a run starts.
        RubricJudgeResponse(scores={"_default": default_score})

    def evaluate(self, request: RubricJudgeRequest) -> RubricJudgeResponse:
        return RubricJudgeResponse(
            scores={
                dimension: self.scores.get(dimension, self.default_score)
                for dimension in request.rubric.dimensions
            },
            feedback=self.feedback,
        )

    def judge(self, request: RubricJudgeRequest) -> RubricJudgeResponse:
        """Alias that keeps the mock convenient outside RubricEvaluator."""
        return self.evaluate(request)
