import json
import math
import re
from typing import Any

from adaptive_router.models import EvaluationResult, EvaluationType, Task
from adaptive_router.output import unwrap_answer


_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric answer")
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        values: list[float] = []
        for item in value.values():
            values.extend(_numeric_values(item))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_numeric_values(item))
        return values
    raise ValueError("expected answer is not numeric")


def _candidate_values(answer: Any) -> list[float]:
    answer = unwrap_answer(answer)
    if isinstance(answer, (dict, list, tuple)):
        return _numeric_values(answer)
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        return [float(answer)]
    text = str(answer)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        return _numeric_values(parsed)
    values = [float(match.replace(",", "")) for match in _NUMBER.findall(text)]
    if not values:
        raise ValueError("answer contains no numeric value")
    return values


def _numeric_match(expected: Any, actual: Any, tolerance: float) -> bool:
    expected = unwrap_answer(expected)
    actual = unwrap_answer(actual)
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(expected) == set(actual)
            and all(_numeric_match(expected[key], actual[key], tolerance) for key in expected)
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(expected) == len(actual)
            and all(_numeric_match(wanted, found, tolerance) for wanted, found in zip(expected, actual))
        )
    if isinstance(expected, bool) or isinstance(actual, bool):
        return False
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


class NumericEvaluator:
    def __init__(self, tolerance: float = 0.0) -> None:
        if tolerance < 0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be a finite non-negative number")
        self.tolerance = tolerance

    def evaluate(self, task: Task, answer: Any) -> EvaluationResult:
        if task.evaluation_type is not EvaluationType.NUMERIC:
            raise ValueError("NumericEvaluator requires a numeric task")
        expected = _numeric_values(task.expected_answer)
        decoded_answer = unwrap_answer(answer)
        try:
            actual = _candidate_values(decoded_answer)
        except ValueError as exc:
            return EvaluationResult(
                quality=0.0,
                passed=False,
                grader_type="numeric",
                feedback=str(exc),
            )
        tolerance = self.tolerance
        configured = task.metadata.get(
            "numeric_tolerance", task.metadata.get("tolerance")
        )
        if configured is not None:
            tolerance = float(configured)
        if isinstance(task.expected_answer, (dict, list, tuple)) and not isinstance(decoded_answer, str):
            correct = _numeric_match(task.expected_answer, decoded_answer, tolerance)
        else:
            correct = len(actual) == len(expected) and all(
                math.isclose(found, wanted, rel_tol=0.0, abs_tol=tolerance)
                for found, wanted in zip(actual, expected)
            )
        return EvaluationResult(
            quality=1.0 if correct else 0.0,
            passed=correct,
            grader_type="numeric",
            feedback=None if correct else f"expected {expected}, received {actual}",
        )
