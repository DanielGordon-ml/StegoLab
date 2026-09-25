"""Fetch job requests, worker documents, progress records and summaries."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from schemas.base import StrictRecord
from schemas.configuration import RequestIdentifier
from schemas.dataset_common import (
    SHA256,
    CounterMap,
    DatasetName,
    DatasetVersionedRecord,
    RelativePath,
    SourceKind,
)
from schemas.dataset_manifest import DatasetSummary
from schemas.dataset_sources import (
    MAXIMUM_DOWNLOAD_BYTES,
    DatasetSourceSpec,
    HuggingFaceSourceSpec,
    SourceName,
)

RunLabel = Annotated[str, Field(pattern=r"^[a-z0-9_]{1,64}$")]
FetchPhase = Literal[
    "resolving", "downloading", "extracting", "preparing", "cleaning", "completed"
]


class DatasetFetchRequest(StrictRecord):
    """Start a fetch that saves raw files under data/ and prepares a revision."""

    client_request_identifier: RequestIdentifier
    operation: Literal["fetch_dataset"] = "fetch_dataset"
    source: DatasetSourceSpec
    source_name: SourceName
    dataset_name: DatasetName
    maximum_images: int = Field(default=200_000, ge=1, le=200_000)
    training_intended: bool = True
    prepare: bool = True

    @model_validator(mode="after")
    def validate_text_sources(self) -> Self:
        """Text corpora are saved as files; they are never prepared as images."""
        if (
            isinstance(self.source, HuggingFaceSourceSpec)
            and self.source.content == "text"
            and self.prepare
        ):
            raise ValueError("Text sources are saved only and cannot be prepared.")
        return self


class DatasetFetchDocument(DatasetVersionedRecord):
    """Private worker input with absolute roots resolved by the backend."""

    schema_version: Literal[1] = 1
    run_label: RunLabel
    source: DatasetSourceSpec
    source_name: SourceName
    dataset_name: DatasetName
    data_root: str = Field(min_length=1, max_length=4096)
    cache_root: str = Field(min_length=1, max_length=4096)
    output_root: str = Field(min_length=1, max_length=4096)
    maximum_images: int = Field(default=200_000, ge=1, le=200_000)
    training_intended: bool = True
    prepare: bool = True
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    resume: bool = True


class DatasetFetchProgress(StrictRecord):
    """Byte and file counters written by the worker while a fetch runs."""

    sequence: int = Field(ge=0)
    phase: FetchPhase
    bytes_received: int = Field(default=0, ge=0)
    bytes_total: int | None = Field(default=None, ge=0, le=MAXIMUM_DOWNLOAD_BYTES)
    files_completed: int = Field(default=0, ge=0)
    files_total: int | None = Field(default=None, ge=0)
    assets_completed: int = Field(default=0, ge=0)
    assets_total: int = Field(default=0, ge=0)
    supports_pause: bool = False
    partial_directories: list[RelativePath] = Field(default_factory=list, max_length=8)
    updated_at: str = Field(min_length=1, max_length=64)


class TextCorpusSummary(StrictRecord):
    """Describe saved text files without quoting their content."""

    files: int = Field(ge=1, le=10_000)
    characters: int = Field(ge=0)
    bytes: int = Field(ge=0)


class DatasetFetchSummary(DatasetVersionedRecord):
    """Public result of a fetch; never a readiness or compatibility claim."""

    schema_version: Literal[1] = 1
    status: Literal["completed", "stopped", "reused"]
    stop_signal: int | None = Field(default=None, ge=1, le=64)
    source_kind: SourceKind
    source_name: SourceName
    reference: str = Field(min_length=1, max_length=4096)
    resolved_revision: str | None = Field(default=None, max_length=128)
    materialization_identity: SHA256
    raw_folder: RelativePath
    bytes_received: int = Field(ge=0)
    bytes_total: int | None = Field(default=None, ge=0)
    assets_completed: int = Field(ge=0)
    assets_total: int = Field(ge=0)
    member_count: int = Field(ge=0, le=200_000)
    rejected_member_count: int = Field(ge=0, le=200_000)
    rejection_reasons: CounterMap = Field(default_factory=dict)
    text_corpus: TextCorpusSummary | None = None
    dataset: DatasetSummary | None = None
    near_duplicate_audit: Literal["not_done"] = "not_done"
    pilot_ready: Literal[False] = False
    warnings: list[str] = Field(default_factory=list, max_length=64)
