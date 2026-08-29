from typing import Any, Protocol

from adaptive_router.models import EvaluationResult, Task


class Evaluator(Protocol):
    def evaluate(self, task: Task, answer: Any) -> EvaluationResult:
        """Evaluate an answer against the grading metadata on task."""
