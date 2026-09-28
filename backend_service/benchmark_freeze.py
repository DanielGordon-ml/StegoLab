"""Freeze the release-benchmark identities once from COCO annotation files."""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from backend_service.dataset_serialization import (
    canonical_json,
    checksum,
    contained_file,
    read_bounded,
)
from backend_service.failures import ApplicationFailure
from schemas.benchmark_identities import (
    BenchmarkAnnotationFile,
    BenchmarkFreezeRequest,
    BenchmarkMember,
    BenchmarkSelectionRule,
    BenchmarkSourceArchive,
    MemberRole,
    ReleaseBenchmarkIdentities,
)

logger = logging.getLogger(__name__)

POLICY_VERSION = "release_benchmark_v1"
VALIDATION_ANNOTATION_FILES = ("captions_val2017.json", "instances_val2017.json")
TRAINING_ANNOTATION_FILE = "captions_train2017.json"
OFFICIAL_IMAGE_COUNTS = (5000, 118_287)
MAXIMUM_ANNOTATION_BYTES = 512 * 1024**2
IDENTITIES_FILE_NAME = "identities.json"
MEMBERS_FILE_NAME = "members.jsonl"
BenchmarkSplit = Literal["val2017", "train2017"]
_MESSAGES: dict[str, tuple[str, int]] = {
    "benchmark_annotations": (
        "The COCO annotation files are missing, unreadable or disagree with each "
        "other. Fetch the official annotations archive again before freezing.",
        422,
    ),
    "benchmark_population": (
        "The training population is smaller than the requested training-source "
        "and reserve counts. Lower the counts or fetch the complete annotations.",
        422,
    ),
    "benchmark_exists": (
        "The frozen benchmark already exists. Validate it instead of freezing again.",
        409,
    ),
    "benchmark_storage": (
        "The frozen benchmark could not be written. Check access and free space, "
        "remove the partial output folder, then freeze again.",
        503,
    ),
    "benchmark_invalid": (
        "The frozen benchmark identities are missing, changed or inconsistent. "
        "Restore the folder from version control or freeze it again.",
        422,
    ),
    "benchmark_mismatch": (
        "The annotations do not re-derive the frozen benchmark members. Check that "
        "the same official annotation files were fetched.",
        422,
    ),
}


@dataclass(frozen=True)
class AnnotatedImage:
    """One image entry exactly as a COCO annotation file declares it."""

    identifier: int
    file_name: str
    width: int
    height: int


def benchmark_failure(code: str) -> ApplicationFailure:
    """Return the fixed plain-language failure for a benchmark failure code."""
    message, status_code = _MESSAGES[code]
    return ApplicationFailure(code, message, status_code)


def selection_key(policy_version: str, seed: int, member_path: str) -> str:
    """Order one training image by hashing the policy, seed and member path."""
    return hashlib.sha256(
        canonical_json([policy_version, seed, member_path])
    ).hexdigest()


def members_checksum(members: list[BenchmarkMember]) -> str:
    """Hash the canonical JSONL member lines exactly as they are written."""
    digest = hashlib.sha256()
    for member in members:
        digest.update(canonical_json(member.model_dump(mode="json")))
    return digest.hexdigest()


def identities_checksum(header: ReleaseBenchmarkIdentities) -> str:
    """Hash the canonical header with its own checksum field left out."""
    values = header.model_dump(mode="json", exclude={"identities_checksum"})
    return checksum(canonical_json(values))


def _annotated_image(item: object) -> AnnotatedImage:
    """Read one image entry, requiring exact integer and string field types."""
    if not isinstance(item, dict):
        raise ValueError("Annotation image entries must be objects.")
    identifier, file_name = item.get("id"), item.get("file_name")
    width, height = item.get("width"), item.get("height")
    if (
        type(identifier) is not int
        or not isinstance(file_name, str)
        or type(width) is not int
        or type(height) is not int
        or file_name != f"{identifier:012d}.jpg"
    ):
        raise ValueError("Annotation image entries must name their identifier.")
    return AnnotatedImage(identifier, file_name, width, height)


def read_annotation_set(
    directory: Path, name: str
) -> tuple[BenchmarkAnnotationFile, tuple[AnnotatedImage, ...]]:
    """Read one bounded annotation file and its unique image entries."""
    try:
        data = read_bounded(contained_file(directory, name), MAXIMUM_ANNOTATION_BYTES)
        document = json.loads(data)
        entries = document.get("images") if isinstance(document, dict) else None
        if not isinstance(entries, list):
            raise ValueError("Annotation files must hold an images array.")
        images = tuple(_annotated_image(item) for item in entries)
        if len({image.identifier for image in images}) != len(images):
            raise ValueError("Annotation image identifiers must be unique.")
        record = BenchmarkAnnotationFile(
            path=name,
            sha256=checksum(data),
            byte_count=len(data),
            image_count=len(images),
        )
    except (OSError, ValueError, MemoryError, RecursionError, ApplicationFailure):
        raise benchmark_failure("benchmark_annotations") from None
    return record, images


def _member(
    image: AnnotatedImage, split: BenchmarkSplit, role: MemberRole, rank: int | None
) -> BenchmarkMember:
    """Build one member from an annotation entry and its assigned role."""
    return BenchmarkMember(
        member_path=f"{split}/{image.file_name}",
        coco_image_identifier=image.identifier,
        declared_width=image.width,
        declared_height=image.height,
        source_split=split,
        role=role,
        rank=rank,
    )


def derive_benchmark(
    annotations_directory: Path,
    *,
    source_archives: list[BenchmarkSourceArchive],
    seed: int,
    training_source_count: int,
    reserve_count: int,
    expect_official_counts: bool,
) -> tuple[ReleaseBenchmarkIdentities, list[BenchmarkMember]]:
    """Derive the header and ordered members from the three annotation files."""
    names = (*VALIDATION_ANNOTATION_FILES, TRAINING_ANNOTATION_FILE)
    files = [read_annotation_set(annotations_directory, name) for name in names]
    (_, captions), (_, instances), (_, training) = files
    counts = (len(captions), len(training))
    if set(captions) != set(instances) or (
        expect_official_counts and counts != OFFICIAL_IMAGE_COUNTS
    ):
        raise benchmark_failure("benchmark_annotations")
    if len(training) < training_source_count + reserve_count:
        raise benchmark_failure("benchmark_population")
    population = sorted(
        training,
        key=lambda image: (
            selection_key(POLICY_VERSION, seed, f"train2017/{image.file_name}"),
            f"train2017/{image.file_name}",
        ),
    )
    try:
        members = sorted(
            (_member(image, "val2017", "validation", None) for image in captions),
            key=lambda member: member.member_path,
        )
        members.extend(
            _member(
                image,
                "train2017",
                "training_source" if rank < training_source_count else "reserve",
                rank,
            )
            for rank, image in enumerate(
                population[: training_source_count + reserve_count]
            )
        )
        header = ReleaseBenchmarkIdentities(
            source_archives=source_archives,
            annotation_files=[record for record, _ in files],
            selection_rule=BenchmarkSelectionRule(
                seed=seed,
                population_count=len(population),
                training_source_count=training_source_count,
                reserve_count=reserve_count,
            ),
            validation_count=len(captions),
            training_source_count=training_source_count,
            reserve_count=reserve_count,
            member_count=len(members),
            members_checksum=members_checksum(members),
            identities_checksum="0" * 64,
        )
    except ValidationError:
        raise benchmark_failure("benchmark_annotations") from None
    header.identities_checksum = identities_checksum(header)
    return header, members


def _write_frozen(
    directory: Path, header: ReleaseBenchmarkIdentities, members: list[BenchmarkMember]
) -> None:
    """Create the output folder and write both files with exclusive creation."""
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise benchmark_failure("benchmark_exists") from None
    except OSError:
        raise benchmark_failure("benchmark_storage") from None
    contents = {
        MEMBERS_FILE_NAME: b"".join(
            canonical_json(member.model_dump(mode="json")) for member in members
        ),
        IDENTITIES_FILE_NAME: canonical_json(header.model_dump(mode="json")),
    }
    try:
        for name, data in contents.items():
            with (directory / name).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
    except OSError:
        raise benchmark_failure("benchmark_storage") from None


def freeze_benchmark(request: BenchmarkFreezeRequest) -> ReleaseBenchmarkIdentities:
    """Freeze the identities into a new folder and return the written header."""
    output = Path(request.output_directory)
    if output.is_symlink() or output.exists():
        raise benchmark_failure("benchmark_exists")
    header, members = derive_benchmark(
        Path(request.annotations_directory),
        source_archives=request.source_archives,
        seed=request.seed,
        training_source_count=request.training_source_count,
        reserve_count=request.reserve_count,
        expect_official_counts=request.expect_official_counts,
    )
    _write_frozen(output, header, members)
    logger.info("benchmark_identities_frozen")
    return header
