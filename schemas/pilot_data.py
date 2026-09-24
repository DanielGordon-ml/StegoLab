"""Frozen pilot cover selection and safe synchronous sampler snapshots."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from schemas.base import StrictRecord
from schemas.dataset_common import SHA256, CounterMap, DatasetVersionedRecord


class PilotCover(StrictRecord):
    """Identify a selected tuning cover without storing an absolute path."""

    source_identity: str = Field(min_length=1, max_length=4096)
    prepared_checksum: SHA256
    width: int = Field(ge=256, le=8192)
    height: int = Field(ge=256, le=8192)


class PilotExcludedCover(PilotCover):
    """Explain why a tuning cover cannot supply the native evaluation crop."""

    reason: Literal["smaller_than_1024"] = "smaller_than_1024"


class PilotSelection(DatasetVersionedRecord):
    """Keep complete split counts and deterministic development-only choices."""

    schema_version: Literal[1] = 1
    policy_version: Literal["pilot_data_v1"] = "pilot_data_v1"
    dataset_revision: SHA256
    records_checksum: SHA256
    split_counts: CounterMap
    training_count: int = Field(ge=1, le=200_000)
    tuning_eligible_count: int = Field(ge=1, le=200_000)
    tuning_selected: list[PilotCover] = Field(min_length=1, max_length=32)
    tuning_excluded: list[PilotExcludedCover] = Field(max_length=200_000)
    training_checksum: SHA256
    tuning_checksum: SHA256
    selection_checksum: SHA256
    gpu_profile_eligible: bool
    gpu_profile_reasons: list[str] = Field(max_length=8)
    pilot_ready: Literal[False] = False


class PilotRandomState(StrictRecord):
    """Represent the standard random generator using bounded primitive values."""

    version: Literal[3] = 3
    values: list[Annotated[int, Field(ge=0, le=2**32 - 1)]] = Field(
        min_length=625, max_length=625
    )

    @model_validator(mode="after")
    def validate_position(self) -> Self:
        """Reject invalid generator cursor values before restoring its state."""
        if self.values[-1] > 624:
            raise ValueError("Random generator position is invalid.")
        return self


class PilotSamplerState(DatasetVersionedRecord):
    """Save only consumed sample order and crop randomness, with no prefetch."""

    schema_version: Literal[1] = 1
    policy_version: Literal["pilot_sampler_v1"] = "pilot_sampler_v1"
    dataset_revision: SHA256
    selection_checksum: SHA256
    seed: int = Field(ge=0, le=2**63 - 1)
    epoch: int = Field(ge=0)
    position: int = Field(ge=0, le=200_000)
    consumed: int = Field(ge=0)
    order: list[int] = Field(min_length=1, max_length=200_000)
    random_state: PilotRandomState


class PilotSample(StrictRecord):
    """Expose the most recently consumed source and crop for reproducibility."""

    source_identity: str = Field(min_length=1, max_length=4096)
    left: int = Field(ge=0, le=8192 - 256)
    top: int = Field(ge=0, le=8192 - 256)
