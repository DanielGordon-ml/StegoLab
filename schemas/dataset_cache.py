"""Immutable download cache entries and their in-flight partial records."""

from typing import Literal

from pydantic import Field

from schemas.base import StrictRecord
from schemas.dataset_common import SHA256, DatasetVersionedRecord, RelativePath
from schemas.dataset_sources import MAXIMUM_DOWNLOAD_BYTES

CacheKind = Literal["hugging_face", "https_archive", "upload"]


class CacheEntryManifest(DatasetVersionedRecord):
    """Freeze what one verified cached file is and where it came from."""

    schema_version: Literal[1] = 1
    kind: CacheKind
    identity: SHA256
    reference: str = Field(min_length=1, max_length=4096)
    revision: str = Field(min_length=1, max_length=128)
    path: RelativePath
    basename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
    size_bytes: int = Field(ge=0, le=MAXIMUM_DOWNLOAD_BYTES)
    sha256: SHA256
    etag: str | None = Field(default=None, max_length=256)
    terms_reference: str = Field(min_length=1, max_length=4096)
    created_at: str = Field(min_length=1, max_length=64)


class CachePartialRecord(StrictRecord):
    """Describe an unfinished download so the owning job can resume or drop it."""

    owner: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    bytes_received: int = Field(ge=0, le=MAXIMUM_DOWNLOAD_BYTES)
    expected_size: int | None = Field(default=None, ge=0, le=MAXIMUM_DOWNLOAD_BYTES)
    expected_sha256: SHA256 | None = None
    etag: str | None = Field(default=None, max_length=256)
    updated_at: str = Field(min_length=1, max_length=64)


class DatasetCacheEntrySummary(StrictRecord):
    """List one cached file with the raw folders and revisions that use it."""

    identity: SHA256
    kind: CacheKind
    reference: str = Field(min_length=1, max_length=4096)
    revision: str = Field(min_length=1, max_length=128)
    path: RelativePath
    size_bytes: int = Field(ge=0)
    created_at: str = Field(min_length=1, max_length=64)
    in_use_by: list[str] = Field(default_factory=list, max_length=200)


class DatasetCacheSummary(StrictRecord):
    """Summarize cache usage without exposing filesystem paths."""

    root_available: bool
    entries: list[DatasetCacheEntrySummary] = Field(
        default_factory=list, max_length=1000
    )
    total_bytes: int = Field(default=0, ge=0)
    unused_bytes: int = Field(default=0, ge=0)
    unused_entries: int = Field(default=0, ge=0)
    partial_bytes: int = Field(default=0, ge=0)
    partial_entries: int = Field(default=0, ge=0)
