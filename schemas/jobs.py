"""Durable job metadata; secret inputs and decoded text are never fields."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field

from schemas.base import StrictRecord
from schemas.configuration import ConfigurationProfile

JobStatus = Literal[
    "queued",
    "running",
    "paused",
    "stopped",
    "completed",
    "cancelled",
    "failed",
    "interrupted",
    "needs_input",
]
JobAction = Literal["pause", "resume", "stop", "cancel", "supply_missing_inputs"]


class JobSnapshot(StrictRecord):
    """Describe a saved job without executing it or exposing secret data."""

    job_identifier: str = Field(min_length=1, max_length=128)
    status: JobStatus
    phase: str = Field(min_length=1, max_length=128)
    configuration: ConfigurationProfile
    available_actions: list[JobAction]
    created_at: AwareDatetime
    updated_at: AwareDatetime
    progress: Annotated[float, Field(ge=0, le=1)] | None = None
    estimated_seconds_remaining: Annotated[int, Field(ge=0)] | None = None
    operation: str | None = None
    experiment_identifier: str | None = None
    frozen_settings: dict[str, object] = Field(default_factory=dict)
    result: dict[str, object] | None = None
    error: dict[str, str] | None = None
    metrics: dict[str, float | int] = Field(default_factory=dict)
    requested_action: JobAction | None = None
    latest_event_identifier: int | None = None


class JobList(StrictRecord):
    """Return saved job snapshots in creation order."""

    items: list[JobSnapshot] = Field(default_factory=list)


class JobEvent(StrictRecord):
    """Define future ordered state notifications with metadata fields only."""

    event_identifier: int = Field(gt=0)
    job_identifier: str = Field(min_length=1, max_length=128)
    status: JobStatus
    phase: str = Field(min_length=1, max_length=128)
    created_at: AwareDatetime
    snapshot: JobSnapshot | None = None
