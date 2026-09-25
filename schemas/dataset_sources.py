"""Remote and uploaded dataset source specifications and inspection results."""

from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import AfterValidator, Field, field_validator, model_validator

from schemas.base import StrictRecord
from schemas.dataset_common import (
    SHA256,
    DatasetName,
    DatasetSplit,
    DatasetVersionedRecord,
    RelativePath,
    SourceKind,
    relative_path,
)

SourceName = DatasetName
UploadReference = Annotated[str, Field(pattern=r"^upload_[0-9a-f]{32}$")]
HubRepository = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"),
]
HubRevision = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")]
ColumnName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
SourceContent = Literal["images", "text"]
UpstreamSplit = Literal["train", "validation", "test"]
SourceAccess = Literal["available", "access_required", "not_found", "unsupported"]
RawFolderState = Literal["available", "reusable", "conflict"]
MAXIMUM_DOWNLOAD_BYTES = 100 * 1024**3
DEFAULT_DOWNLOAD_BYTES = 25 * 1024**3


def https_url(value: str) -> str:
    """Accept only plain https addresses without credentials, ports or fragments."""
    parts = urlsplit(value)
    if (
        len(value) > 4096
        or parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or parts.port not in (None, 443)
        or any(character.isspace() for character in value)
    ):
        raise ValueError("Source addresses must be plain https without credentials.")
    return value


HttpsUrl = Annotated[str, AfterValidator(https_url)]


class SourceSpecificationBase(StrictRecord):
    """Fields shared by every remote or uploaded dataset source."""

    archive_splits: dict[str, DatasetSplit] | None = Field(default=None, max_length=64)
    maximum_download_bytes: int = Field(
        default=DEFAULT_DOWNLOAD_BYTES, ge=1, le=MAXIMUM_DOWNLOAD_BYTES
    )
    terms_reference: str = Field(min_length=1, max_length=4096)

    @field_validator("archive_splits")
    @classmethod
    def validate_archive_split_keys(
        cls, value: dict[str, DatasetSplit] | None
    ) -> dict[str, DatasetSplit] | None:
        """Require safe relative member folders as split mapping keys."""
        for key in value or {}:
            relative_path(key)
        return value


class HuggingFaceSourceSpec(SourceSpecificationBase):
    """Pin a Hugging Face dataset repository and select files inside it."""

    source_kind: Literal["hugging_face"] = "hugging_face"
    repository: HubRepository
    revision: HubRevision = "main"
    content: SourceContent = "images"
    path_prefix: RelativePath | None = None
    file_names: list[RelativePath] | None = Field(default=None, max_length=200)
    split: UpstreamSplit | None = None
    image_column: ColumnName | None = None
    text_column: ColumnName | None = None
    maximum_files: int = Field(default=200, ge=1, le=10_000)

    @model_validator(mode="after")
    def validate_file_names(self) -> Self:
        """Keep explicit file lists nonempty and unique."""
        if self.file_names is not None and (
            not self.file_names or len(set(self.file_names)) != len(self.file_names)
        ):
            raise ValueError("Selected file names must be nonempty and unique.")
        return self


class HttpsArchiveSourceSpec(SourceSpecificationBase):
    """Download one zip or tar archive from a plain https address."""

    source_kind: Literal["https_archive"] = "https_archive"
    url: HttpsUrl
    expected_sha256: SHA256 | None = None


class UploadSourceSpec(SourceSpecificationBase):
    """Use an archive that the browser uploaded in parts."""

    source_kind: Literal["upload"] = "upload"
    upload_identifier: UploadReference
    terms_reference: str = Field(default="not_reviewed", min_length=1, max_length=4096)


DatasetSourceSpec = Annotated[
    HuggingFaceSourceSpec | HttpsArchiveSourceSpec | UploadSourceSpec,
    Field(discriminator="source_kind"),
]


class DatasetInspectionRequest(StrictRecord):
    """Ask what a source contains before any bytes are downloaded."""

    source: DatasetSourceSpec
    source_name: SourceName | None = None


class PlannedAssetRecord(StrictRecord):
    """Describe one file that a fetch would download or reuse."""

    path: RelativePath
    size_bytes: int | None = Field(default=None, ge=0, le=MAXIMUM_DOWNLOAD_BYTES)
    sha256: SHA256 | None = None


class DatasetInspection(StrictRecord):
    """Report a resolved source honestly, including unknown sizes."""

    source_kind: SourceKind
    reference: str = Field(min_length=1, max_length=4096)
    requested_revision: str | None = Field(default=None, max_length=128)
    resolved_revision: str | None = Field(default=None, max_length=128)
    suggested_source_name: SourceName
    access: SourceAccess
    access_guidance: str | None = Field(default=None, max_length=1024)
    content: SourceContent
    asset_count: int = Field(ge=0, le=10_000)
    assets: list[PlannedAssetRecord] = Field(default_factory=list, max_length=200)
    download_bytes: int | None = Field(default=None, ge=0, le=MAXIMUM_DOWNLOAD_BYTES)
    materialized_bytes: int | None = Field(default=None, ge=0)
    cached_bytes: int = Field(default=0, ge=0)
    supports_pause: bool
    declared_splits: bool
    materialization_identity: SHA256
    raw_folder: RawFolderState | None = None
    free_disk_bytes: int = Field(ge=0)
    required_free_bytes: int | None = Field(default=None, ge=0)
    disk_sufficient: bool | None = None
    server_token_configured: bool
    warnings: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def validate_disk_claim(self) -> Self:
        """Only claim disk sufficiency when the required bytes are known."""
        if (self.required_free_bytes is None) != (self.disk_sufficient is None):
            raise ValueError("Disk sufficiency requires a known byte estimate.")
        return self


class SourceMarker(DatasetVersionedRecord):
    """Provenance stored beside materialized raw files under data/<source_name>/."""

    schema_version: Literal[1] = 1
    source_kind: SourceKind
    source_name: SourceName
    reference: str = Field(min_length=1, max_length=4096)
    source_url: str = Field(min_length=1, max_length=4096)
    terms_reference: str = Field(min_length=1, max_length=4096)
    resolved_revision: str = Field(min_length=1, max_length=128)
    materialization_identity: SHA256
    content: SourceContent
    assets: list[PlannedAssetRecord] = Field(default_factory=list, max_length=10_000)
    member_count: int = Field(ge=0, le=200_000)
    rejected_member_count: int = Field(ge=0, le=200_000)
    split_mapping: dict[str, DatasetSplit] = Field(default_factory=dict)
    declared_splits: bool
    created_at: str = Field(min_length=1, max_length=64)
    completed: bool
