"""Read-only local preparation checks and honest pending GPU evidence."""

from typing import Literal

from pydantic import Field, field_validator

from schemas.base import StrictRecord


class PilotPreflightRequest(StrictRecord):
    """Locate prepared data and host state without launching or training anything."""

    schema_version: Literal[2] = 2
    dataset_directory: str = Field(min_length=1, max_length=4096)
    output_root: str = Field(default=".", min_length=1, max_length=4096)
    device: Literal["cpu", "cuda"] = "cpu"
    session_identifier: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_integer_version(cls, value: object) -> object:
        """Keep the preflight version as strict as the training request version."""
        if type(value) is not int:
            raise ValueError("The schema version requires a whole number.")
        return value


class PilotReadinessCheck(StrictRecord):
    """Keep local results separate from checks requiring real GPU execution."""

    name: str
    status: Literal["passed_locally", "failed", "not_run"]
    detail: str


class PilotPreflightReport(StrictRecord):
    """Describe preparation only; never promote data or release capabilities."""

    schema_version: Literal[2] = 2
    preparation_passed: bool
    pilot_ready: Literal[False] = False
    gpu_checks_run: Literal[False] = False
    dataset_revision: str | None = None
    selection_checksum: str | None = None
    split_counts: dict[str, int] = Field(default_factory=dict)
    training_images: int = Field(default=0, ge=0)
    tuning_images: int = Field(default=0, ge=0)
    excluded_tuning_images: int = Field(default=0, ge=0)
    disk_free_bytes: int = Field(ge=0)
    required_free_reserve_bytes: int = Field(ge=0)
    remaining_gpu_seconds: float | None = Field(default=None, ge=0, le=86400)
    environment: dict[str, str]
    checks: list[PilotReadinessCheck]
