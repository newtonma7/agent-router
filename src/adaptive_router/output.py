"""Machine-readable output contracts shared by agents and evaluators."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


_MACHINE_SYSTEM_PROMPT = (
    "You are a task-solving agent. Follow the user task exactly. "
    "Return exactly one JSON object with one top-level key, `answer`. "
    "The answer value must match the requested answer type. Do not include "
    "Markdown, code fences, explanations, or any other top-level keys."
)
_RUBRIC_SYSTEM_PROMPT = (
    "You are a task-solving agent. Follow the user task exactly and return "
    "the answer as plain natural language. Do not add a JSON wrapper or grading metadata."
)


def _value_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, (int, float)):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {str(key): _value_schema(item) for key, item in value.items()},
            "required": [str(key) for key in value],
            "additionalProperties": False,
        }
    if isinstance(value, list):
        item_schema = _value_schema(value[0]) if value else {}
        return {
            "type": "array",
            "items": item_schema,
            "minItems": len(value),
            "maxItems": len(value),
        }
    if value is None:
        return {"type": "null"}
    return {}


def task_output_contract(task: Any) -> tuple[str, dict[str, Any] | None]:
    """Return the system instruction and OpenAI schema for one task."""
    raw_evaluation_type = getattr(task, "evaluation_type", None)
    if raw_evaluation_type is None:
        return "", None
    evaluation_type = getattr(raw_evaluation_type, "value", raw_evaluation_type)
    if str(evaluation_type).lower() == "rubric":
        return _RUBRIC_SYSTEM_PROMPT, None

    expected = getattr(task, "expected_answer", None)
    kind = str(evaluation_type).lower()
    guidance = {
        "numeric": "For numeric tasks, include only the final numeric value or values in `answer`; do not include working.",
        "exact": "For exact tasks, put only the final name, letter, or short answer in `answer`.",
        "structured": "For structured tasks, put the requested object or array in `answer` exactly as specified.",
    }.get(kind, "Put the final answer in `answer`.")
    schema = {
        "type": "object",
        "properties": {"answer": _value_schema(expected)},
        "required": ["answer"],
        "additionalProperties": False,
    }
    return f"{_MACHINE_SYSTEM_PROMPT} {guidance}", {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_answer",
            "strict": True,
            "schema": schema,
        },
    }


def unwrap_answer(value: Any) -> Any:
    """Decode JSON output and remove the standard machine-output wrapper."""
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value
    if isinstance(parsed, Mapping) and set(parsed) == {"answer"}:
        return parsed["answer"]
    return parsed


def rubric_judge_response_format(dimensions: Mapping[str, Any]) -> dict[str, Any]:
    """Build the strict JSON schema for a rubric judge response."""
    names = [str(name) for name in dimensions]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rubric_judge_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "object",
                        "properties": {
                            name: {"type": "integer", "minimum": 0, "maximum": 4}
                            for name in names
                        },
                        "required": names,
                        "additionalProperties": False,
                    },
                    "feedback": {"type": ["string", "null"]},
                },
                "required": ["scores", "feedback"],
                "additionalProperties": False,
            },
        },
    }


__all__ = ["rubric_judge_response_format", "task_output_contract", "unwrap_answer"]
