"""Strict source requests and prepared image provenance."""

from typing import Literal, Self

from pydantic import Field, model_validator

from schemas.base import StrictRecord
from schemas.dataset_common import (
    SHA256,
    DatasetName,
    DatasetSplit,
    DatasetVersionedRecord,
    RelativePath,
)

SOURCE_URL = "https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html"


class DatasetPreparationRequest(DatasetVersionedRecord):
    """Freeze a local source, policy, and optional explicitly named subset."""

    schema_version: Literal[1] = 1
    dataset_name: DatasetName = "uhd_iqa"
    source_directory: str = Field(min_length=1, max_length=4096)
    metadata_file: RelativePath | None = "uhd-iqa-metadata.csv"
    output_root: str = Field(default="datasets", min_length=1, max_length=4096)
    source_kind: Literal["uhd_iqa", "local"] = "uhd_iqa"
    policy_version: Literal["dataset_v1"] = "dataset_v1"
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    selection_name: DatasetName | None = None
    selection: list[RelativePath] | None = Field(default=None, max_length=200_000)
    source_url: str = Field(default=SOURCE_URL, max_length=4096)
    terms_reference: str = Field(default=SOURCE_URL, max_length=4096)

    @model_validator(mode="before")
    @classmethod
    def default_local_provenance(cls, value: object) -> object:
        """Avoid attributing unlabelled local sources to the UHD-IQA database."""
        if isinstance(value, dict) and value.get("source_kind") == "local":
            value = dict(value)
            value.setdefault("source_url", "local")
            value.setdefault("terms_reference", "not_reviewed")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Require explicit names and unique safe paths for reduced coverage."""
        if self.source_kind == "uhd_iqa" and self.metadata_file is None:
            raise ValueError("UHD-IQA preparation requires source metadata.")
        if (self.selection is None) != (self.selection_name is None):
            raise ValueError("A selection requires both its name and source paths.")
        if self.selection is not None and (
            not self.selection or len(set(self.selection)) != len(self.selection)
        ):
            raise ValueError("Selection paths must be nonempty and unique.")
        return self


class DatasetImageRecord(StrictRecord):
    """Describe original bytes and the immutable prepared pixels."""

    source_path: RelativePath
    prepared_path: RelativePath
    source_identity: str | None = Field(default=None, min_length=1, max_length=4096)
    upstream_split: Literal["training", "validation", "test"] | None = None
    subset: str | None = Field(default=None, max_length=4096)
    assigned_split: DatasetSplit | None = None
    duplicate_group: str = Field(default="", pattern=r"^([a-f0-9]{64})?$")
    split_group: str = Field(default="", pattern=r"^([a-f0-9]{64})?$")
    representative_source_path: RelativePath | None = None
    source_checksum: SHA256
    prepared_checksum: SHA256
    rgb_checksum: SHA256
    source_bytes: int = Field(ge=1, le=50 * 1024**2)
    prepared_bytes: int = Field(ge=1, le=128 * 1024**2)
    source_width: int = Field(ge=1, le=8192)
    source_height: int = Field(ge=1, le=8192)
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    source_format: Literal["JPEG", "PNG"]
    source_mode: Literal["L", "RGB", "RGBA"]
    mode: Literal["RGB", "RGBA"]
    orientation: int = Field(ge=1, le=8)
    color_policy: Literal["icc_to_srgb", "declared_srgb", "assumed_srgb"]
    grayscale_converted: bool
    eligible: bool

    @model_validator(mode="after")
    def validate_image_policy(self) -> Self:
        """Prevent provenance records from contradicting the frozen image policy."""
        if max(self.width * self.height, self.source_width * self.source_height) > 32e6:
            raise ValueError("Dataset image exceeds the pixel limit.")
        expected_size = (self.source_width, self.source_height)
        if self.orientation in (5, 6, 7, 8):
            expected_size = expected_size[::-1]
        if (self.width, self.height) != expected_size:
            raise ValueError("Dataset preparation must preserve oriented dimensions.")
        if self.eligible != (self.width >= 256 and self.height >= 256):
            raise ValueError("Eligibility does not match the training crop policy.")
        if self.grayscale_converted != (self.source_mode == "L"):
            raise ValueError("Grayscale conversion does not match the source mode.")
        if self.mode != ("RGBA" if self.source_mode == "RGBA" else "RGB"):
            raise ValueError("Prepared mode does not preserve supported channels.")
        if self.source_format == "JPEG" and self.source_mode == "RGBA":
            raise ValueError("RGBA JPEG is unsupported.")
        return self


class DatasetRejection(StrictRecord):
    """Record a rejected candidate without private exception text."""

    source_path: RelativePath
    source_checksum: SHA256 | None = None
    upstream_split: Literal["training", "validation", "test"] | None = None
    reason: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    source_bytes: int = Field(default=0, ge=0, le=100 * 1024**3)
