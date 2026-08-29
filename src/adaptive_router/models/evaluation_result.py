from pydantic import BaseModel, ConfigDict, Field


class EvaluationResult(BaseModel):
    """The stable result returned by every evaluator."""

    model_config = ConfigDict(extra="forbid")

    quality: float = Field(ge=0.0, le=1.0)
    passed: bool
    grader_type: str = Field(min_length=1)
    component_scores: dict[str, float] = Field(default_factory=dict)
    feedback: str | None = None
