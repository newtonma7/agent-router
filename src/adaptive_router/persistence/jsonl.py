from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict") and callable(value.dict):
        return _jsonable(value.dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value


@dataclass
class RunRecord:
    task_id: str
    policy: str
    context: tuple[float, ...] | list[float]
    action: str
    strategy: str | None = None
    category: str | None = None
    answer: Any = None
    evaluation: Any = None
    quality: float | None = None
    passed: bool | None = None
    cost_usd: float | None = None
    latency_seconds: float | None = None
    normalized_cost: float | None = None
    normalized_latency: float | None = None
    reward: float | None = None
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str = ""
    configuration: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: int | None = None
    regret: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = _jsonable(asdict(self))
        if not data["config"] and data["configuration"]:
            data["config"] = data["configuration"]
        if not data["configuration"] and data["config"]:
            data["configuration"] = data["config"]
        return data


class JSONLRecorder:
    """Append complete records; a failed write is deliberately propagated."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: RunRecord | Mapping[str, Any] | Any) -> dict[str, Any]:
        data = record.to_dict() if hasattr(record, "to_dict") else _jsonable(record)
        if not isinstance(data, dict):
            raise TypeError("a run record must serialize to an object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
        return data

    record = append
    write = append

    def __iter__(self):
        return iter(load_records(self.path))


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


read_jsonl = load_records
