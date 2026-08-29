import re
from typing import Any

from adaptive_router.models import EvaluationResult, EvaluationType, Task


def normalize_answer(value: Any) -> str:
    """Normalize harmless formatting differences without changing meaning."""
    text = str(value).strip().strip("`*_#")
    return " ".join(text.casefold().split())


class ExactEvaluator:
    def evaluate(self, task: Task, answer: Any) -> EvaluationResult:
        if task.evaluation_type is not EvaluationType.EXACT:
            raise ValueError("ExactEvaluator requires an exact-answer task")

        expected = normalize_answer(task.expected_answer)
        actual_text = normalize_answer(answer)
        match = re.search(
            r"(?:^|[.!?\n])\s*(?:(?:the )?(?:answer|conclusion)\s*(?:is|:)\s*)?"
            r"(yes|no)\b",
            actual_text,
        )
        if expected in {"yes", "no"}:
            correct = match is not None and match.group(1) == expected
        else:
            conclusion = re.split(r"[.!?\n]", actual_text, maxsplit=1)[0].strip()
            conclusion = re.sub(
                r"^(?:the )?answer\s*(?:is|:)\s*", "", conclusion
            )
            correct = conclusion == expected or conclusion.startswith(expected + " ")
        return EvaluationResult(
            quality=1.0 if correct else 0.0,
            passed=correct,
            grader_type="exact",
            feedback=None if correct else f"expected {expected!r}",
        )
