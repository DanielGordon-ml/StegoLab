"""Explicit later GPU readiness requests and honest per-check evidence."""

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from schemas.base import StrictRecord
from schemas.dataset_common import SHA256
from schemas.pilot_budget import SessionIdentifier


class GpuReadinessRequest(StrictRecord):
    """Identify an existing host session and optional frozen learned checkpoint."""

    schema_version: Literal[2] = 2
    dataset_directory: str = Field(min_length=1, max_length=4096)
    output_root: str = Field(default=".", min_length=1, max_length=4096)
    experiment_identifier: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    session_identifier: SessionIdentifier | None = None
    learned_checkpoint: str | None = Field(default=None, min_length=1, max_length=4096)
    learned_experiment_identifier: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$"
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_version_integer(cls, value: object) -> object:
        """Reject floating-point equivalents at the version boundary."""
        if type(value) is not int:
            raise ValueError("The schema version must be an integer.")
        return value

    @model_validator(mode="after")
    def require_checkpoint_identity(self) -> Self:
        """Require the original experiment identity with any learned weights."""
        if (self.learned_checkpoint is None) != (
            self.learned_experiment_identifier is None
        ):
            raise ValueError("A learned checkpoint requires its experiment identifier.")
        return self


class GpuReadinessCheck(StrictRecord):
    """Distinguish unattempted checks from real hardware passes or failures."""

    name: str
    status: Literal["passed_locally", "passed_on_gpu", "failed", "not_run"]
    detail: str
    elapsed_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    error_code: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class GpuReadinessReport(StrictRecord):
    """Keep infrastructure evidence separate from release or dataset promotion."""

    schema_version: Literal[2] = 2
    experiment_identifier: str
    execution: Literal["check_only", "explicit_gpu_run"]
    readiness_passed: bool = False
    pilot_ready: Literal[False] = False
    dataset_revision: SHA256 | None = None
    selection_checksum: SHA256 | None = None
    report_directory: str
    environment: dict[str, str] = Field(default_factory=dict)
    checks: list[GpuReadinessCheck]


class GpuResumeEvidence(StrictRecord):
    """Record exact continuation plus measurements for a later profile choice."""

    physical_batch_size: Literal[4, 8]
    effective_batch_size: Literal[16] = 16
    completed_optimizer_updates: Literal[2] = 2
    consumed_samples: Literal[32] = 32
    uninterrupted_training_seconds: float = Field(ge=0, allow_inf_nan=False)
    resumed_update_seconds: float = Field(ge=0, allow_inf_nan=False)
    peak_device_memory_bytes: int = Field(ge=0)
    checkpoint: str
    exact_model_optimizer_random_sampler_match: Literal[True] = True
    exact_next_payload_crop_match: Literal[True] = True
