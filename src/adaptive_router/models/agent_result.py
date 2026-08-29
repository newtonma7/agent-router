from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import AgentStrategy


class AgentResult(BaseModel):
    """Common measured result produced by every agent strategy."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    strategy: AgentStrategy
    answer: Any
    latency_seconds: float = Field(ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    tool_calls: int = Field(default=0, ge=0)
    error: str | None = None
