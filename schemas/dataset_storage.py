"""Storage totals for downloads, raw source folders and prepared datasets."""

from pydantic import Field

from schemas.base import StrictRecord


class DatasetStorageSummary(StrictRecord):
    """Show where dataset bytes live and whether unused downloads can go."""

    cache_bytes: int = Field(ge=0)
    cache_entries: int = Field(ge=0)
    cache_unused_bytes: int = Field(ge=0)
    cache_unused_entries: int = Field(ge=0)
    raw_source_bytes: int = Field(ge=0)
    raw_source_folders: int = Field(ge=0)
    prepared_bytes: int = Field(ge=0)
    prepared_revisions: int = Field(ge=0)
    free_disk_bytes: int = Field(ge=0)
    minimum_free_bytes: int = Field(ge=0)
    active_fetch_jobs: int = Field(ge=0)
    cleanup_available: bool
