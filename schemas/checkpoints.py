"""Validated checkpoint metadata, independent of executable model state."""

from typing import Literal

from pydantic import Field, model_validator

from schemas.base import StrictRecord


class CheckpointMetrics(StrictRecord):
    """Rank measured candidates by exact recovery, image error, then structure."""

    exact_message_recovery: float = Field(ge=0, le=1, allow_inf_nan=False)
    median_psnr: float = Field(allow_inf_nan=False)
    median_ssim: float = Field(ge=-1, le=1, allow_inf_nan=False)


class CheckpointSummary(StrictRecord):
    """Describe a verified immutable state file without exposing its contents."""

    schema_version: Literal[1] = 1
    checkpoint_identifier: str = Field(pattern=r"^checkpoint_[0-9]+_[a-f0-9]{32}$")
    global_step: int = Field(ge=0)
    created_at: str = Field(min_length=1, max_length=64)
    configuration_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    state_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    state_bytes: int = Field(gt=0, le=1024**3)
    state_filename: Literal["state.pt"] = "state.pt"
    configuration: dict[str, object]
    identities: dict[str, str] = Field(min_length=1)
    metrics: CheckpointMetrics | None = None
    pinned: bool = False


class CheckpointIndex(StrictRecord):
    """Publish the latest recovery candidate and retained immutable states."""

    schema_version: Literal[1] = 1
    latest_identifier: str | None = None
    checkpoints: list[CheckpointSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identifiers(self) -> "CheckpointIndex":
        """Reject duplicate entries and a latest pointer to unavailable state."""
        identifiers = [item.checkpoint_identifier for item in self.checkpoints]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Checkpoint identifiers must be unique.")
        if (
            self.latest_identifier is not None
            and self.latest_identifier not in identifiers
        ):
            raise ValueError("The latest checkpoint must be in the index.")
        if self.checkpoints and self.latest_identifier is None:
            raise ValueError("A populated index requires a latest checkpoint.")
        return self
