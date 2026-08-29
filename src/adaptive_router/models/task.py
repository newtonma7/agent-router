from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import EvaluationType, TaskCategory
from .rubric import EvaluationRubric


class Task(BaseModel):
    """A validated benchmark or inference task."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    category: TaskCategory
    evaluation_type: EvaluationType
    expected_answer: Any | None = None
    rubric: EvaluationRubric | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_grading_contract(self) -> "Task":
        expected_evaluation = {
            TaskCategory.ARITHMETIC: EvaluationType.NUMERIC,
            TaskCategory.REASONING: EvaluationType.EXACT,
            TaskCategory.EXPLANATION: EvaluationType.RUBRIC,
            TaskCategory.EXTRACTION: EvaluationType.STRUCTURED,
        }[self.category]
        if self.evaluation_type is not expected_evaluation:
            raise ValueError(
                f"{self.category.value} tasks require {expected_evaluation.value} evaluation"
            )

        needs_expected = self.evaluation_type in {
            EvaluationType.NUMERIC,
            EvaluationType.EXACT,
            EvaluationType.STRUCTURED,
        }
        if needs_expected and self.expected_answer is None:
            raise ValueError(
                f"{self.evaluation_type.value} tasks require expected_answer"
            )
        if self.evaluation_type is EvaluationType.RUBRIC:
            if self.rubric is None:
                raise ValueError("rubric tasks require rubric")
            if self.expected_answer is not None:
                raise ValueError("rubric tasks must not define expected_answer")
        elif self.rubric is not None:
            raise ValueError("only rubric tasks may define rubric")
        return self


# Keep the enums discoverable from the task module, where callers naturally look.
__all__ = ["EvaluationRubric", "EvaluationType", "Task", "TaskCategory"]
