"""Near-duplicate audit reports kept outside immutable dataset revisions."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from schemas.base import StrictRecord
from schemas.dataset_common import (
    SHA256,
    CounterMap,
    DatasetName,
    DatasetSplit,
    DatasetVersionedRecord,
)

AuditIdentifier = Annotated[str, Field(pattern=r"^[a-f0-9]{12}_[0-9]{8}T[0-9]{6}Z$")]
AuditThreshold = Literal[0, 4, 8, 12]
AUDIT_THRESHOLDS: tuple[AuditThreshold, ...] = (0, 4, 8, 12)
MAXIMUM_REPORTED_PAIRS = 2000


class NearDuplicateAuditRequest(DatasetVersionedRecord):
    """Audit one prepared revision, optionally against the frozen benchmark."""

    schema_version: Literal[1] = 1
    dataset_directory: str = Field(min_length=1, max_length=4096)
    output_root: str | None = Field(default=None, min_length=1, max_length=4096)
    benchmark_identities_directory: str | None = Field(
        default=None, min_length=1, max_length=4096
    )
    benchmark_dataset_directory: str | None = Field(
        default=None, min_length=1, max_length=4096
    )
    limit: int | None = Field(default=None, ge=1, le=200_000)


class NearDuplicatePair(StrictRecord):
    """One pair of images whose perceptual hashes are close."""

    first_identity: str = Field(min_length=1, max_length=4096)
    first_split: DatasetSplit | None = None
    first_source: Literal["dataset", "benchmark"]
    second_identity: str = Field(min_length=1, max_length=4096)
    second_split: DatasetSplit | None = None
    second_source: Literal["dataset", "benchmark"]
    distance: int = Field(ge=0, le=64)


class NearDuplicateThresholdCounts(StrictRecord):
    """Pair counts at one Hamming distance level."""

    threshold: AuditThreshold
    within_split: CounterMap = Field(default_factory=dict)
    cross_split: CounterMap = Field(default_factory=dict)
    cross_split_total: int = Field(ge=0)
    benchmark_by_split: CounterMap | None = None
    benchmark_total: int | None = Field(default=None, ge=0)


class BenchmarkComparison(StrictRecord):
    """State whether benchmark pixels were available for comparison."""

    identities_checksum: SHA256
    status: Literal["pixels_unavailable", "compared"]
    benchmark_revision: SHA256 | None = None
    member_count: int = Field(ge=0, le=300_000)
    compared_members: int = Field(ge=0, le=300_000)
    missing_members: int = Field(ge=0, le=300_000)


class NearDuplicateAuditReport(DatasetVersionedRecord):
    """Evidence report bound to one revision; never a readiness decision."""

    schema_version: Literal[1] = 1
    method_version: Literal["phash_dct_32_v1"] = "phash_dct_32_v1"
    audit_identifier: AuditIdentifier
    dataset_name: DatasetName
    dataset_revision: SHA256
    records_checksum: SHA256
    hashed_images: int = Field(ge=0, le=200_000)
    hashed_by_split: CounterMap = Field(default_factory=dict)
    limited: bool
    thresholds: list[NearDuplicateThresholdCounts] = Field(min_length=4, max_length=4)
    pairs: list[NearDuplicatePair] = Field(
        default_factory=list, max_length=MAXIMUM_REPORTED_PAIRS
    )
    pairs_total_at_maximum_threshold: int = Field(ge=0)
    pairs_truncated: bool
    benchmark: BenchmarkComparison | None = None
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    peak_process_mebibytes: int = Field(ge=0)
    environment: dict[str, str] = Field(default_factory=dict)
    pilot_ready: Literal[False] = False
    report_checksum: SHA256

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        """Require the four fixed levels in ascending order."""
        if tuple(item.threshold for item in self.thresholds) != AUDIT_THRESHOLDS:
            raise ValueError("Audit reports list the levels 0, 4, 8 and 12 in order.")
        if self.pairs_truncated != (
            self.pairs_total_at_maximum_threshold > len(self.pairs)
        ):
            raise ValueError("Pair truncation must match the reported counts.")
        return self
