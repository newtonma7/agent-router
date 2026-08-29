import json
import re
from typing import Any

from adaptive_router.models import EvaluationResult, EvaluationType, Task
from adaptive_router.output import unwrap_answer


def _parse_json(answer: Any) -> Any:
    if not isinstance(answer, str):
        return answer
    text = answer.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    return unwrap_answer(json.loads(text))


def _leaves(value: Any, path: tuple[str | int, ...] = ()) -> list[tuple[tuple[str | int, ...], Any]]:
    if isinstance(value, dict):
        leaves: list[tuple[tuple[str | int, ...], Any]] = []
        for key, item in value.items():
            leaves.extend(_leaves(item, path + (key,)))
        return leaves
    if isinstance(value, list):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_leaves(item, path + (index,)))
        return leaves
    return [(path, value)]


def _same_shape(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(expected) == set(actual)
            and all(_same_shape(expected[key], actual[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(_same_shape(wanted, found) for wanted, found in zip(expected, actual))
        )
    return not isinstance(actual, (dict, list))


def _same_value(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip() == actual.strip()
    return type(expected) is type(actual) and expected == actual


class StructuredEvaluator:
    def evaluate(self, task: Task, answer: Any) -> EvaluationResult:
        if task.evaluation_type is not EvaluationType.STRUCTURED:
            raise ValueError("StructuredEvaluator requires a structured task")
        expected_leaves = dict(_leaves(task.expected_answer))
        try:
            parsed = _parse_json(answer)
            actual_leaves = dict(_leaves(parsed))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return EvaluationResult(
                quality=0.0,
                passed=False,
                grader_type="structured",
                component_scores={"correct_fields": 0.0, "total_fields": float(len(expected_leaves))},
                feedback=f"invalid JSON: {exc}",
            )
        correct = sum(
            path in actual_leaves and _same_value(value, actual_leaves[path])
            for path, value in expected_leaves.items()
        )
        total = len(expected_leaves)
        quality = correct / total if total else 0.0
        passed = correct == total and _same_shape(task.expected_answer, parsed)
        return EvaluationResult(
            quality=quality,
            passed=passed,
            grader_type="structured",
            component_scores={"correct_fields": float(correct), "total_fields": float(total)},
            feedback=None if passed else f"{correct} of {total} required fields matched",
        )
