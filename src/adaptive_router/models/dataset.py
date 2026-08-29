import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .task import Task


class SeedDataset(BaseModel):
    """The frozen, versioned twelve-task development benchmark."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    tasks: list[Task] = Field(min_length=12, max_length=12)

    @field_validator("tasks")
    @classmethod
    def validate_task_ids(cls, tasks: list[Task]) -> list[Task]:
        ids = [task.id for task in tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("seed task ids must be unique")
        return tasks


def load_seed_dataset(path: str | Path) -> SeedDataset:
    return SeedDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))
