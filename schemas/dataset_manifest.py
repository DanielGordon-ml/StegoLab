"""Immutable dataset identity and safe inspection summaries."""

from typing import Literal

from pydantic import Field

from schemas.dataset_common import (
    SHA256,
    CounterMap,
    DatasetName,
    DatasetSplit,
    DatasetVersionedRecord,
    RelativePath,
    SourceKind,
)


class DatasetManifest(DatasetVersionedRecord):
    """Freeze all decisions required for source-independent dataset reuse."""

    schema_version: Literal[1] = 1
    policy_version: Literal["dataset_v1"] = "dataset_v1"
    revision: SHA256
    dataset_name: DatasetName
    source_kind: SourceKind
    source_url: str
    terms_reference: str
    source_provenance: dict[str, str]
    metadata_checksum: SHA256 | None
    split_mapping: dict[str, DatasetSplit]
    seed: int = Field(ge=0, le=2**63 - 1)
    selection_name: DatasetName | None
    selection: list[RelativePath] | None
    expected_images: int | None = Field(ge=0, le=200_000)
    discovered_images: int = Field(ge=0, le=200_000)
    selected_images: int = Field(ge=0, le=200_000)
    expected_by_split: CounterMap
    discovered_by_split: CounterMap
    selected_by_split: CounterMap
    full_coverage: bool
    records_checksum: SHA256
    rejections_checksum: SHA256
    accepted_count: int = Field(ge=1, le=200_000)
    rejection_count: int = Field(ge=0, le=200_000)
    eligible_count: int = Field(ge=1, le=200_000)
    ineligible_count: int = Field(ge=0, le=200_000)
    duplicate_count: int = Field(ge=0, le=200_000)
    unique_eligible_count: int = Field(ge=1, le=200_000)
    split_counts: CounterMap
    eligible_by_split: CounterMap
    unique_eligible_by_split: CounterMap
    rejection_reasons: CounterMap
    source_bytes: int = Field(ge=1, le=100 * 1024**3)
    prepared_bytes: int = Field(ge=1, le=100 * 1024**3)
    pilot_ready: Literal[False] = False


class DatasetSummary(DatasetVersionedRecord):
    """Report safe counts while distinguishing inspection from verification."""

    schema_version: Literal[1] = 1
    dataset_name: DatasetName
    revision: SHA256
    completion: Literal["complete"] = "complete"
    integrity: Literal["not_checked", "verified"]
    reused: bool = False
    full_coverage: bool
    expected_images: int | None
    discovered_images: int
    selected_images: int
    accepted_count: int
    rejection_count: int
    eligible_count: int
    ineligible_count: int
    duplicate_count: int
    unique_eligible_count: int
    split_counts: CounterMap
    eligible_by_split: CounterMap
    unique_eligible_by_split: CounterMap
    rejection_reasons: CounterMap
    source_bytes: int
    prepared_bytes: int
    pilot_ready: Literal[False] = False
    warnings: list[str] = Field(default_factory=list)
