"""Browser workflow requests using registered references instead of paths."""

from typing import Literal

from pydantic import Field

from schemas.base import StrictRecord
from schemas.configuration import RequestIdentifier

WorkflowOperation = Literal["train", "evaluate", "export", "prepare_dataset"]


class WorkflowRequest(StrictRecord):
    """Resolve a bounded local operation on the server's original workspace."""

    client_request_identifier: RequestIdentifier
    operation: WorkflowOperation
    experiment_identifier: str = Field(
        default="cpu_experiment", pattern=r"^[a-z][a-z0-9_]{0,63}$"
    )
    dataset_identifier: str | None = Field(default=None, max_length=128)
    checkpoint_identifier: str | None = Field(default=None, max_length=128)
    cpu_threads: int = Field(default=4, ge=1, le=16)
    stop_after_step: int = Field(default=1000, ge=1, le=1000)
    source_identifier: str | None = Field(default=None, max_length=128)
    source_subdirectory: str = Field(default="", max_length=4096)
    dataset_name: str = Field(default="local_images", pattern=r"^[a-z][a-z0-9_]{0,63}$")
    export_name: str = Field(
        default="001V", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
    )
    lightweight: bool = False


class WorkflowActionRequest(StrictRecord):
    """Make retries of supported actions safe across browser reloads."""

    client_request_identifier: RequestIdentifier
    action: Literal["stop", "cancel", "pause", "resume"]


class TrainingPreflight(StrictRecord):
    """Explain eligibility without creating files or reserving experiment time."""

    allowed: bool
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resolved_settings: dict[str, object] = Field(default_factory=dict)
    remaining_budget_seconds: float = Field(default=0, ge=0)
    remaining_experiment_slots: int = Field(default=0, ge=0)
