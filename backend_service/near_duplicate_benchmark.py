"""Hash prepared records and compare them with the frozen release benchmark."""

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from backend_service.benchmark_validation import load_benchmark_identities
from backend_service.dataset_serialization import contained_file
from backend_service.dataset_validation import validated_records
from backend_service.failures import ApplicationFailure
from backend_service.near_duplicate_hash import (
    MAXIMUM_THRESHOLD,
    audit_failure,
    hash_prepared_image,
    pairs_within,
)
from backend_service.pilot_data import source_identity
from schemas.datasets import DatasetImageRecord, DatasetManifest
from schemas.near_duplicate_audit import (
    AUDIT_THRESHOLDS,
    BenchmarkComparison,
    NearDuplicateAuditRequest,
    NearDuplicatePair,
)

logger = logging.getLogger(__name__)
Progress = Callable[[str], None] | None
PROGRESS_INTERVAL = 500


@dataclass(frozen=True)
class HashedImages:
    """Keep hashes beside the identities and split names of their records."""

    records: tuple[DatasetImageRecord, ...]
    hashes: NDArray[np.uint64]
    identities: tuple[str, ...]
    splits: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    """Carry the comparison block with its pairs and per-level split counts."""

    comparison: BenchmarkComparison
    pairs: list[NearDuplicatePair]
    counts: dict[int, Counter[str]]


def notify(progress: Progress, event: str) -> None:
    """Record one fixed event name in the module log and the optional callback."""
    logger.info(event)
    if progress is not None:
        progress(event)


def load_revision(
    directory: Path, code: str
) -> tuple[DatasetManifest, list[DatasetImageRecord]]:
    """Validate a prepared revision's metadata without decoding any pixels."""
    try:
        return validated_records(directory)
    except (OSError, ValueError, MemoryError, ApplicationFailure):
        raise audit_failure(code) from None


def hash_images(
    directory: Path, records: Sequence[DatasetImageRecord], *, progress: Progress
) -> HashedImages:
    """Hash records one image at a time, reusing the hash of a shared file."""
    hashes = np.zeros(len(records), dtype=np.uint64)
    known: dict[str, int] = {}
    for index, record in enumerate(records):
        value = known.get(record.prepared_path)
        if value is None:
            try:
                path = contained_file(directory, record.prepared_path)
            except ValueError:
                raise audit_failure("audit_integrity") from None
            value = hash_prepared_image(path, record)
            known[record.prepared_path] = value
        hashes[index] = value
        if (index + 1) % PROGRESS_INTERVAL == 0:
            notify(progress, "near_duplicate_hashing_progress")
    return HashedImages(
        tuple(records),
        hashes,
        tuple(source_identity(record) for record in records),
        tuple(record.assigned_split or "unassigned" for record in records),
    )


def member_path(record: DatasetImageRecord) -> str:
    """Read the member path from an identity, whole when it has no source name.

    Fetched archives write ``<source name>:<member path>``; a locally prepared
    folder has no identity and falls back to its source path, which is already
    the member path when the folder kept the COCO layout.
    """
    head, separator, tail = source_identity(record).partition(":")
    return tail if separator else head


def compare_benchmark(
    request: NearDuplicateAuditRequest,
    dataset: HashedImages,
    audited_revision: str,
    *,
    progress: Progress,
) -> BenchmarkResult:
    """Bind the audit to the identities and compare pixels when a revision exists."""
    header, members = load_benchmark_identities(
        Path(request.benchmark_identities_directory or "")
    )
    counts: dict[int, Counter[str]] = {level: Counter() for level in AUDIT_THRESHOLDS}
    block: dict[str, object] = {
        "identities_checksum": header.identities_checksum,
        "member_count": header.member_count,
        "status": "pixels_unavailable",
        "compared_members": 0,
        "missing_members": header.member_count,
    }
    if request.benchmark_dataset_directory is None:
        return BenchmarkResult(BenchmarkComparison.model_validate(block), [], counts)
    directory = Path(request.benchmark_dataset_directory)
    manifest, records = load_revision(directory, "audit_benchmark_dataset_invalid")
    if manifest.revision == audited_revision:
        raise audit_failure("audit_same_revision")
    member_paths = {member.member_path for member in members}
    matched: dict[str, DatasetImageRecord] = {}
    for record in records:
        path = member_path(record)
        if path in member_paths:
            matched.setdefault(path, record)
    if not matched:
        raise audit_failure("audit_benchmark_unmatched")
    benchmark = hash_images(directory, list(matched.values()), progress=progress)
    paths = list(matched)
    pairs = []
    for first, second, distance in pairs_within(
        dataset.hashes, benchmark.hashes, threshold=MAXIMUM_THRESHOLD, same_set=False
    ):
        for threshold in AUDIT_THRESHOLDS:
            if distance <= threshold:
                counts[threshold][dataset.splits[first]] += 1
        pairs.append(
            NearDuplicatePair(
                first_identity=dataset.identities[first],
                first_split=dataset.records[first].assigned_split,
                first_source="dataset",
                second_identity=paths[second],
                second_split=None,
                second_source="benchmark",
                distance=distance,
            )
        )
    block.update(
        status="compared",
        benchmark_revision=manifest.revision,
        compared_members=len(matched),
        missing_members=header.member_count - len(matched),
    )
    return BenchmarkResult(BenchmarkComparison.model_validate(block), pairs, counts)
