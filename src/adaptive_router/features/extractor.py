"""Extract the small, explicit context available before an action is run."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adaptive_router.models.task import Task

CATEGORY_ORDER = ("arithmetic", "reasoning", "explanation", "extraction")
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\.\d+|\d+(?:,\d{3})*)(?:%\b)?")
_FIELD = re.compile(
    r"(?:fields?|keys?|columns?)\s*(?:are|:|=)?\s*"
    r"([A-Za-z_][\w]*(?:\s*,\s*(?:and\s+)?[A-Za-z_][\w]*)+)",
    re.I,
)
_OBJECT_FIELDS = re.compile(
    r"(?:objects?|records?)\s+(?:contain|include|have)\s+"
    r"([A-Za-z_][\w]*(?:\s*,\s*(?:and\s+)?[A-Za-z_][\w]*)+)",
    re.I,
)
_EXTRACT_FIELDS = re.compile(
    r"(?:extract|return)\s+([A-Za-z_][\w]*(?:\s*,\s*(?:and\s+)?[A-Za-z_][\w]*)+)\s+from\b",
    re.I,
)


def _value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _task_value(task: Any, name: str, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        return task.get(name, default)
    return getattr(task, name, default)


def _list_count(value: str) -> int:
    return sum(1 for item in value.split(",") if item.strip().removeprefix("and ").strip())


def _requested_fields(prompt: str) -> int:
    """Count fields explicitly requested in a prompt, without reading answers."""
    quoted = re.findall(r"[`\"]([A-Za-z_][\w -]*)[`\"]", prompt)
    listed = sum(
        _list_count(match)
        for match in (*_FIELD.findall(prompt), *_OBJECT_FIELDS.findall(prompt), *_EXTRACT_FIELDS.findall(prompt))
    )
    return max(len(quoted), listed)


class FeatureExtractor:
    """Create a fixed-order vector from task data that precedes execution.

    The vector is category one-hot, normalized prompt length, number presence,
    normalized numeric count, and normalized requested-field count.
    """

    def __init__(self, max_prompt_length: int = 1000, max_numeric_count: int = 10, max_field_count: int = 10):
        if max_prompt_length <= 0 or max_numeric_count <= 0 or max_field_count <= 0:
            raise ValueError("feature normalization limits must be positive")
        self.max_prompt_length = max_prompt_length
        self.max_numeric_count = max_numeric_count
        self.max_field_count = max_field_count

    def extract(self, task: Any) -> tuple[float, ...]:
        prompt = str(_task_value(task, "prompt", ""))
        category = _value(_task_value(task, "category", ""))
        numbers = _NUMBER.findall(prompt)
        return (
            *(1.0 if category == expected else 0.0 for expected in CATEGORY_ORDER),
            min(len(prompt) / self.max_prompt_length, 1.0),
            float(bool(numbers)),
            min(len(numbers) / self.max_numeric_count, 1.0),
            min(_requested_fields(prompt) / self.max_field_count, 1.0),
        )

    __call__ = extract


def extract_features(task: Any) -> tuple[float, ...]:
    """Convenience function using the default normalization constants."""
    return FeatureExtractor().extract(task)


extract_feature_vector = extract_features
