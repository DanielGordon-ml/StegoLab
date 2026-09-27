"""Frozen release-benchmark identities built from COCO annotations and archives."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from schemas.base import StrictRecord
from schemas.dataset_common import SHA256, DatasetVersionedRecord, RelativePath
from schemas.dataset_sources import HubRepository

CommitRevision = Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
MemberPath = Annotated[str, Field(pattern=r"^(val2017|train2017)/[0-9]{12}\.jpg$")]
MemberRole = Literal["validation", "training_source", "reserve"]
MAXIMUM_ARCHIVE_BYTES = 100 * 1024**3


class BenchmarkSourceArchive(StrictRecord):
    """Pin one archive by repository commit, path, size and checksum."""

    repository: HubRepository
    repository_kind: Literal["huggingface_dataset"] = "huggingface_dataset"
    revision: CommitRevision
    archive_path: RelativePath
    archive_sha256: SHA256
    archive_bytes: int = Field(ge=1, le=MAXIMUM_ARCHIVE_BYTES)


class BenchmarkAnnotationFile(StrictRecord):
    """Record the annotation file that supplied identities and dimensions."""

    path: RelativePath
    sha256: SHA256
    byte_count: int = Field(ge=1, le=1024**3)
    image_count: int = Field(ge=0, le=1_000_000)


class BenchmarkSelectionRule(StrictRecord):
    """State the deterministic ordering used to choose training-source members."""

    policy_version: Literal["release_benchmark_v1"] = "release_benchmark_v1"
    seed: int = Field(ge=0, le=2**63 - 1)
    ordering: Literal["sha256_of_canonical_json_policy_seed_member_path"] = (
        "sha256_of_canonical_json_policy_seed_member_path"
    )
    population_split: Literal["train2017"] = "train2017"
    population_count: int = Field(ge=1, le=1_000_000)
    training_source_count: int = Field(ge=1, le=100_000)
    reserve_count: int = Field(ge=0, le=100_000)


class BenchmarkMember(StrictRecord):
    """Identify one benchmark image by archive member path and COCO identifier."""

    member_path: MemberPath
    coco_image_identifier: int = Field(ge=1, le=10**9)
    declared_width: int = Field(ge=1, le=8192)
    declared_height: int = Field(ge=1, le=8192)
    source_split: Literal["val2017", "train2017"]
    role: MemberRole
    rank: int | None = Field(default=None, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        """Keep roles, splits, ranks and file names consistent."""
        validation = self.role == "validation"
        if validation != (self.source_split == "val2017"):
            raise ValueError("Validation members come from val2017 only.")
        if validation != (self.rank is None):
            raise ValueError("Only ranked members carry a rank.")
        expected = f"{self.source_split}/{self.coco_image_identifier:012d}.jpg"
        if self.member_path != expected:
            raise ValueError("Member paths must match the COCO identifier.")
        return self


class ReleaseBenchmarkIdentities(DatasetVersionedRecord):
    """Frozen header for the 10,000-image release benchmark identities."""

    schema_version: Literal[1] = 1
    benchmark_name: Literal["release_benchmark_v1"] = "release_benchmark_v1"
    source_archives: list[BenchmarkSourceArchive] = Field(min_length=1, max_length=8)
    annotation_files: list[BenchmarkAnnotationFile] = Field(min_length=1, max_length=8)
    selection_rule: BenchmarkSelectionRule
    validation_count: int = Field(ge=1, le=100_000)
    training_source_count: int = Field(ge=1, le=100_000)
    reserve_count: int = Field(ge=0, le=100_000)
    member_count: int = Field(ge=1, le=300_000)
    members_checksum: SHA256
    identities_checksum: SHA256
    preparation_policy: Literal["not_frozen"] = "not_frozen"
    pilot_ready: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Keep the header counts consistent with the selection rule."""
        total = self.validation_count + self.training_source_count + self.reserve_count
        if self.member_count != total:
            raise ValueError("Member count must equal the sum of the roles.")
        if (
            self.selection_rule.training_source_count != self.training_source_count
            or self.selection_rule.reserve_count != self.reserve_count
        ):
            raise ValueError("Selection rule counts disagree with the header.")
        return self


class BenchmarkFreezeRequest(DatasetVersionedRecord):
    """Build the frozen identities once from fetched annotations."""

    schema_version: Literal[1] = 1
    annotations_directory: str = Field(
        default="data/coco2017/annotations", min_length=1, max_length=4096
    )
    output_directory: str = Field(
        default="docs/benchmarks/release_benchmark_v1", min_length=1, max_length=4096
    )
    source_archives: list[BenchmarkSourceArchive] = Field(min_length=1, max_length=8)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    training_source_count: int = Field(default=5000, ge=1, le=100_000)
    reserve_count: int = Field(default=1000, ge=0, le=100_000)
    expect_official_counts: bool = True


class BenchmarkValidationReport(DatasetVersionedRecord):
    """Confirm a frozen identities folder without changing it."""

    schema_version: Literal[1] = 1
    benchmark_name: Literal["release_benchmark_v1"] = "release_benchmark_v1"
    identities_checksum: SHA256
    members_checksum: SHA256
    validation_count: int = Field(ge=1, le=100_000)
    training_source_count: int = Field(ge=1, le=100_000)
    reserve_count: int = Field(ge=0, le=100_000)
    member_count: int = Field(ge=1, le=300_000)
    integrity: Literal["verified", "verified_with_annotations"]
    pilot_ready: Literal[False] = False
