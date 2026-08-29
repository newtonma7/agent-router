"""Typed settings loaded from environment variables."""

from __future__ import annotations

import math
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


class SettingsError(ValueError):
    """Raised when environment-backed router settings are invalid."""


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read the small ``KEY=value`` subset used by the example file."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if not key or not key.isidentifier():
            continue
        raw_value = raw_value.strip()
        if raw_value.startswith(("'", '"')):
            try:
                parsed = shlex.split(raw_value, comments=True)
            except ValueError as exc:
                raise SettingsError(f"invalid .env value for {key}") from exc
            value = parsed[0] if parsed else ""
        else:
            value = raw_value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


@dataclass(frozen=True, repr=False)
class Settings:
    """Runtime settings with deterministic, mock-first defaults."""

    mock_mode: bool = True
    openai_api_key: str | None = field(default=None, repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    request_timeout_seconds: float = 60.0
    direct_model: str = "gpt-4o-mini"
    strong_model: str = "gpt-4o"
    tool_model: str = "gpt-4o-mini"
    direct_input_cost_per_1k_tokens: float = 0.0
    direct_output_cost_per_1k_tokens: float = 0.0
    strong_input_cost_per_1k_tokens: float = 0.0
    strong_output_cost_per_1k_tokens: float = 0.0
    tool_input_cost_per_1k_tokens: float = 0.0
    tool_output_cost_per_1k_tokens: float = 0.0
    max_tool_calls: int = 3
    cost_penalty: float = 0.0
    latency_penalty: float = 0.0
    epsilon: float = 0.1
    ucb_confidence: float = 2.0
    linucb_alpha: float = 1.0
    persistence_path: str = "runs.jsonl"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Load settings from ``.env`` and the process environment.

        Explicit process environment values win over values in ``.env``.
        Passing ``environ`` keeps callers and tests fully isolated from files.
        """
        values = {}
        if environ is None:
            values.update(_read_dotenv(Path(".env")))
            values.update(os.environ)
        else:
            values.update(environ)

        def get(name: str, *aliases: str, default: str | None = None) -> str | None:
            for key in (f"ADAPTIVE_ROUTER_{name}", *aliases):
                if key in values:
                    return values[key]
            return default

        def text(name: str, *aliases: str, default: str) -> str:
            value = get(name, *aliases, default=default)
            assert value is not None
            return value

        def boolean(name: str, *aliases: str, default: bool) -> bool:
            value = get(name, *aliases, default=str(default))
            assert value is not None
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise SettingsError(f"{name} must be a boolean")

        def integer(name: str, *aliases: str, default: int, minimum: int | None = None) -> int:
            raw = get(name, *aliases, default=str(default))
            assert raw is not None
            try:
                parsed = int(raw)
            except ValueError as exc:
                raise SettingsError(f"{name} must be an integer") from exc
            if minimum is not None and parsed < minimum:
                raise SettingsError(f"{name} must be >= {minimum}")
            return parsed

        def number(
            name: str,
            *aliases: str,
            default: float,
            minimum: float | None = None,
        ) -> float:
            raw = get(name, *aliases, default=str(default))
            assert raw is not None
            try:
                parsed = float(raw)
            except ValueError as exc:
                raise SettingsError(f"{name} must be a number") from exc
            if not math.isfinite(parsed):
                raise SettingsError(f"{name} must be finite")
            if minimum is not None and parsed < minimum:
                raise SettingsError(f"{name} must be >= {minimum}")
            return parsed

        mock_mode = boolean("MOCK_MODE", default=True)
        api_key = get("OPENAI_API_KEY", "OPENAI_API_KEY")
        if not mock_mode and not api_key:
            raise SettingsError("OPENAI_API_KEY is required when MOCK_MODE is false")

        return cls(
            mock_mode=mock_mode,
            openai_api_key=api_key,
            openai_base_url=text("OPENAI_BASE_URL", "OPENAI_BASE_URL", default=cls.openai_base_url),
            request_timeout_seconds=number("REQUEST_TIMEOUT_SECONDS", default=60.0, minimum=0.001),
            direct_model=text("DIRECT_MODEL", "DIRECT_MODEL", "OPENAI_DIRECT_MODEL", default=cls.direct_model),
            strong_model=text("STRONG_MODEL", "STRONG_MODEL", "OPENAI_STRONG_MODEL", default=cls.strong_model),
            tool_model=text("TOOL_MODEL", "TOOL_MODEL", "OPENAI_TOOL_MODEL", default=cls.tool_model),
            direct_input_cost_per_1k_tokens=number("DIRECT_INPUT_COST_PER_1K_TOKENS", "DIRECT_INPUT_PRICE_PER_1K", default=0.0, minimum=0.0),
            direct_output_cost_per_1k_tokens=number("DIRECT_OUTPUT_COST_PER_1K_TOKENS", "DIRECT_OUTPUT_PRICE_PER_1K", default=0.0, minimum=0.0),
            strong_input_cost_per_1k_tokens=number("STRONG_INPUT_COST_PER_1K_TOKENS", "STRONG_INPUT_PRICE_PER_1K", default=0.0, minimum=0.0),
            strong_output_cost_per_1k_tokens=number("STRONG_OUTPUT_COST_PER_1K_TOKENS", "STRONG_OUTPUT_PRICE_PER_1K", default=0.0, minimum=0.0),
            tool_input_cost_per_1k_tokens=number("TOOL_INPUT_COST_PER_1K_TOKENS", "TOOL_INPUT_PRICE_PER_1K", default=0.0, minimum=0.0),
            tool_output_cost_per_1k_tokens=number("TOOL_OUTPUT_COST_PER_1K_TOKENS", "TOOL_OUTPUT_PRICE_PER_1K", default=0.0, minimum=0.0),
            max_tool_calls=integer("MAX_TOOL_CALLS", default=3, minimum=0),
            cost_penalty=number("COST_PENALTY", default=0.0, minimum=0.0),
            latency_penalty=number("LATENCY_PENALTY", default=0.0, minimum=0.0),
            epsilon=number("EPSILON", default=0.1, minimum=0.0),
            ucb_confidence=number("UCB_CONFIDENCE", default=2.0, minimum=0.0),
            linucb_alpha=number("LINUCB_ALPHA", default=1.0, minimum=0.0),
            persistence_path=text("PERSISTENCE_PATH", default="runs.jsonl"),
        )

    def __repr__(self) -> str:
        return (
            "Settings("
            f"mock_mode={self.mock_mode!r}, openai_base_url={self.openai_base_url!r}, "
            f"direct_model={self.direct_model!r}, strong_model={self.strong_model!r}, "
            f"tool_model={self.tool_model!r}, max_tool_calls={self.max_tool_calls!r})"
        )
