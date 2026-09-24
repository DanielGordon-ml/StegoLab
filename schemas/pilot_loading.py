"""Strict evidence from a bounded CPU-only pilot image loading probe."""

from typing import Literal

from pydantic import Field

from schemas.dataset_common import SHA256, DatasetVersionedRecord


class PilotLoadingReport(DatasetVersionedRecord):
    """Record loading cost without scoring images or changing model parameters."""

    schema_version: Literal[1] = 1
    operation: Literal["pilot_loading_probe"] = "pilot_loading_probe"
    status: Literal["passed", "failed"]
    error_code: str | None = None
    dataset_revision: SHA256 | None = None
    selection_checksum: SHA256 | None = None
    device: Literal["cpu"] = "cpu"
    threads: int = Field(ge=1, le=4)
    requested_count: int = Field(ge=1, le=16)
    consumed_count: int = Field(ge=0, le=16)
    crop_shapes: list[tuple[int, int, int, int]] = Field(max_length=4)
    source_identities: list[str] = Field(max_length=16)
    deadline_seconds: float = Field(gt=0, le=120)
    elapsed_seconds: float = Field(ge=0)
    peak_process_mebibytes: float = Field(ge=0)
    immutable_file_metadata_unchanged: bool
    platform: str
    python: str
