"""Audit one prepared revision for near-duplicate images and write the evidence."""

import platform
import resource
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import PIL

from backend_service.dataset_serialization import canonical_json, checksum, read_bounded
from backend_service.failures import ApplicationFailure
from backend_service.near_duplicate_benchmark import (
    HashedImages,
    compare_benchmark,
    hash_images,
    load_revision,
    notify,
)
from backend_service.near_duplicate_hash import (
    MAXIMUM_THRESHOLD,
    audit_failure,
    pairs_within,
)
from backend_service.near_duplicate_output import audit_output_path as audit_output_path
from backend_service.near_duplicate_output import resolve_output_root, write_report
from schemas.dataset_common import DATASET_SPLITS
from schemas.near_duplicate_audit import (
    AUDIT_THRESHOLDS,
    MAXIMUM_REPORTED_PAIRS,
    AuditThreshold,
    NearDuplicateAuditReport,
    NearDuplicateAuditRequest,
    NearDuplicatePair,
    NearDuplicateThresholdCounts,
)

MAXIMUM_REPORT_BYTES = 16 * 1024**2
CROSS_SPLIT_KEYS = ("train_tuning", "train_held_out", "tuning_held_out")
SPLIT_ORDER: dict[str, int] = {name: index for index, name in enumerate(DATASET_SPLITS)}


def current_time() -> datetime:
    """Return the moment that names the audit; tests replace it for fixed names."""
    return datetime.now(UTC)


def audit_identifier(revision: str, now: datetime) -> str:
    """Name an audit by the revision prefix and its UTC start time."""
    return f"{revision[:12]}_{now:%Y%m%dT%H%M%SZ}"


def report_checksum(report: NearDuplicateAuditReport) -> str:
    """Hash every report field except the checksum itself."""
    return checksum(
        canonical_json(report.model_dump(mode="json", exclude={"report_checksum"}))
    )


def load_audit_report(path: Path) -> NearDuplicateAuditReport:
    """Read a bounded report and prove it is unchanged since it was written."""
    try:
        data = read_bounded(path, MAXIMUM_REPORT_BYTES)
        report = NearDuplicateAuditReport.model_validate_json(data)
    except (OSError, ValueError, MemoryError, ApplicationFailure):
        raise audit_failure("audit_report_invalid") from None
    if report_checksum(report) != report.report_checksum:
        raise audit_failure("audit_report_invalid")
    return report


def _cross_key(first: str, second: str) -> str:
    """Name a split pair in the fixed train, tuning, held_out order."""
    ordered = sorted(
        (first, second),
        key=lambda name: (SPLIT_ORDER.get(name, len(SPLIT_ORDER)), name),
    )
    return f"{ordered[0]}_{ordered[1]}"


def _dataset_pairs(
    hashed: HashedImages,
) -> tuple[list[NearDuplicatePair], dict[int, Counter[str]], dict[int, Counter[str]]]:
    """List close pairs once and count them within and across splits per level."""
    within: dict[int, Counter[str]] = {level: Counter() for level in AUDIT_THRESHOLDS}
    cross: dict[int, Counter[str]] = {level: Counter() for level in AUDIT_THRESHOLDS}
    pairs = []
    for first, second, distance in pairs_within(
        hashed.hashes, hashed.hashes, threshold=MAXIMUM_THRESHOLD, same_set=True
    ):
        first_split, second_split = hashed.splits[first], hashed.splits[second]
        for threshold in AUDIT_THRESHOLDS:
            if distance > threshold:
                continue
            if first_split == second_split:
                within[threshold][first_split] += 1
            else:
                cross[threshold][_cross_key(first_split, second_split)] += 1
        pairs.append(
            NearDuplicatePair(
                first_identity=hashed.identities[first],
                first_split=hashed.records[first].assigned_split,
                first_source="dataset",
                second_identity=hashed.identities[second],
                second_split=hashed.records[second].assigned_split,
                second_source="dataset",
                distance=distance,
            )
        )
    return pairs, within, cross


def _threshold_counts(
    threshold: AuditThreshold,
    within: Counter[str],
    cross: Counter[str],
    benchmark: Counter[str] | None,
    splits: list[str],
) -> NearDuplicateThresholdCounts:
    """Fill one level with zero defaults for every split and split pair."""
    keys = [*CROSS_SPLIT_KEYS, *sorted(set(cross) - set(CROSS_SPLIT_KEYS))]
    cross_split = {key: cross.get(key, 0) for key in keys}
    return NearDuplicateThresholdCounts(
        threshold=threshold,
        within_split={name: within.get(name, 0) for name in splits},
        cross_split=cross_split,
        cross_split_total=sum(cross_split.values()),
        benchmark_by_split=None
        if benchmark is None
        else {name: benchmark.get(name, 0) for name in splits},
        benchmark_total=None if benchmark is None else sum(benchmark.values()),
    )


def _peak_mebibytes() -> int:
    """Read the process peak memory the way dataset preparation records it."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024**2 if sys.platform == "darwin" else 1024))


def _environment() -> dict[str, str]:
    """Name the software versions that influence the hashes."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }


def audit_near_duplicates(
    request: NearDuplicateAuditRequest, *, progress: Callable[[str], None] | None = None
) -> NearDuplicateAuditReport:
    """Hash every representative image, count close pairs and save the report."""
    started = time.perf_counter()
    now = current_time()
    if (
        request.benchmark_dataset_directory is not None
        and request.benchmark_identities_directory is None
    ):
        raise audit_failure("audit_benchmark_identities_required")
    directory = Path(request.dataset_directory)
    manifest, records = load_revision(directory, "audit_dataset_invalid")
    root = resolve_output_root(request)
    notify(progress, "near_duplicate_audit_started")
    representatives = sorted(
        (
            record
            for record in records
            if record.source_path == record.representative_source_path
        ),
        key=lambda record: record.source_path,
    )
    limited = request.limit is not None and request.limit < len(representatives)
    if request.limit is not None:
        representatives = representatives[: request.limit]
    hashed = hash_images(directory, representatives, progress=progress)
    pairs, within, cross = _dataset_pairs(hashed)
    benchmark = None
    benchmark_counts: dict[int, Counter[str]] = {}
    if request.benchmark_identities_directory is not None:
        result = compare_benchmark(
            request, hashed, manifest.revision, progress=progress
        )
        benchmark, benchmark_counts = result.comparison, result.counts
        pairs.extend(result.pairs)
    splits = [*DATASET_SPLITS, *sorted(set(hashed.splits) - set(DATASET_SPLITS))]
    pairs.sort(
        key=lambda pair: (pair.distance, pair.first_identity, pair.second_identity)
    )
    report = NearDuplicateAuditReport(
        audit_identifier=audit_identifier(manifest.revision, now),
        dataset_name=manifest.dataset_name,
        dataset_revision=manifest.revision,
        records_checksum=manifest.records_checksum,
        hashed_images=len(representatives),
        hashed_by_split={name: hashed.splits.count(name) for name in splits},
        limited=limited,
        thresholds=[
            _threshold_counts(
                threshold,
                within[threshold],
                cross[threshold],
                None if benchmark is None else benchmark_counts[threshold],
                splits,
            )
            for threshold in AUDIT_THRESHOLDS
        ],
        pairs=pairs[:MAXIMUM_REPORTED_PAIRS],
        pairs_total_at_maximum_threshold=len(pairs),
        pairs_truncated=len(pairs) > MAXIMUM_REPORTED_PAIRS,
        benchmark=benchmark,
        elapsed_seconds=round(time.perf_counter() - started, 4),
        peak_process_mebibytes=_peak_mebibytes(),
        environment=_environment(),
        report_checksum="0" * 64,
    )
    report.report_checksum = report_checksum(report)
    write_report(root, report)
    notify(progress, "near_duplicate_audit_completed")
    return report
