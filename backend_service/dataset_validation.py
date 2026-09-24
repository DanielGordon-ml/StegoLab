"""Source-independent verification of immutable dataset revisions."""

import hashlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from backend_service.dataset_groups import assign_groups
from backend_service.dataset_manifest import manifest_revision, record_counts, summarize
from backend_service.dataset_serialization import (
    canonical_json,
    contained_file,
    file_checksum,
    read_bounded,
    read_records,
)
from backend_service.failures import ApplicationFailure
from schemas.dataset_common import SPLIT_MAPPING
from schemas.datasets import (
    DatasetImageRecord,
    DatasetManifest,
    DatasetRejection,
    DatasetSummary,
)


def load_manifest(directory: Path) -> DatasetManifest:
    """Parse one bounded manifest while rejecting noncanonical or forged identity."""
    data = read_bounded(contained_file(directory, "manifest.json"))
    manifest = DatasetManifest.model_validate_json(data)
    values = manifest.model_dump(mode="json")
    if canonical_json(values) != data or manifest_revision(values) != manifest.revision:
        raise ValueError("Manifest identity or serialization is inconsistent.")
    return manifest


def inspect_dataset(directory: Path) -> DatasetSummary:
    """Read manifest counts without claiming prepared files have been verified."""
    try:
        return summarize(load_manifest(directory), "not_checked")
    except (OSError, ValueError, MemoryError, ApplicationFailure):
        raise _integrity_failure() from None


def validate_dataset(
    directory: Path, *, require_revision_name: bool = True
) -> DatasetSummary:
    """Verify checksums, pixel data, groups, splits, paths, counts, and revision."""
    manifest, _ = load_validated_dataset(
        directory, require_revision_name=require_revision_name
    )
    return summarize(manifest, "verified")


def load_validated_dataset(
    directory: Path,
    *,
    require_revision_name: bool = True,
    check_progress: Callable[[], None] | None = None,
) -> tuple[DatasetManifest, list[DatasetImageRecord]]:
    """Return the same frozen records whose outputs and metadata were validated."""
    if check_progress is not None:
        check_progress()
    try:
        manifest, records = validated_records(directory, require_revision_name)
    except (OSError, ValueError, MemoryError, ApplicationFailure):
        raise _integrity_failure() from None
    from backend_service.dataset_image import validate_dataset_png

    checked: set[str] = set()
    for record in records:
        if record.prepared_path in checked:
            continue
        if check_progress is not None:
            check_progress()
        checked.add(record.prepared_path)
        try:
            path = contained_file(directory, record.prepared_path)
            digest, size = file_checksum(path, 128 * 1024**2)
            if (digest, size) != (record.prepared_checksum, record.prepared_bytes):
                raise ValueError("A prepared file has changed.")
            pixels = validate_dataset_png(path)
            if (
                pixels.width,
                pixels.height,
                pixels.mode,
                pixels.rgb_checksum,
                pixels.prepared_checksum,
                pixels.prepared_bytes,
            ) != (
                record.width,
                record.height,
                record.mode,
                record.rgb_checksum,
                record.prepared_checksum,
                record.prepared_bytes,
            ):
                raise ValueError("Prepared pixels do not match frozen records.")
        except (OSError, ValueError, MemoryError, ApplicationFailure):
            raise _integrity_failure() from None
    if check_progress is not None:
        check_progress()
    return manifest, records


def validated_records(
    directory: Path, require_revision_name: bool = True
) -> tuple[DatasetManifest, list[DatasetImageRecord]]:
    """Validate canonical metadata and logical invariants before decoding pixels."""
    manifest = load_manifest(directory)
    if require_revision_name and directory.name != manifest.revision:
        raise ValueError("Revision directory and manifest identity disagree.")
    records = read_records(directory, "records.jsonl", DatasetImageRecord)
    rejections = read_records(directory, "rejections.jsonl", DatasetRejection)
    for rows, expected in (
        (records, manifest.records_checksum),
        (rejections, manifest.rejections_checksum),
    ):
        digest = hashlib.sha256()
        for row in rows:
            digest.update(canonical_json(row.model_dump(mode="json")))
        if digest.hexdigest() != expected:
            raise ValueError("Parsed dataset record checksums disagree.")
    all_paths = [record.source_path for record in records + rejections]
    prepared_paths = [record.prepared_path for record in records]
    if (
        len(all_paths) > 200_000
        or len(all_paths) != len(set(all_paths))
        or any(
            not path.startswith("images/") or not path.endswith(".png")
            for path in prepared_paths
        )
        or rejections != sorted(rejections, key=lambda record: record.source_path)
    ):
        raise ValueError("Dataset paths are inconsistent or duplicated.")
    _validate_prepared_paths(directory, records)
    if assign_groups(records, manifest.source_kind, manifest.seed) != records:
        raise ValueError("Frozen duplicate groups or splits are inconsistent.")
    values = manifest.model_dump(mode="json")
    if any(
        values[key] != value
        for key, value in record_counts(records, rejections).items()
    ):
        raise ValueError("Manifest counts disagree with frozen records.")
    _validate_coverage(manifest, records, all_paths, rejections)
    return manifest, records


def _validate_prepared_paths(
    directory: Path, records: list[DatasetImageRecord]
) -> None:
    """Permit shared byte-identical outputs and reject untracked image files."""
    if {path.name for path in directory.iterdir()} != {
        "images",
        "manifest.json",
        "records.jsonl",
        "rejections.jsonl",
    }:
        raise ValueError("Revision contains untracked files or directories.")
    signatures: dict[str, tuple[object, ...]] = {}
    source_signatures: dict[str, tuple[object, ...]] = {}
    for record in records:
        signature = (
            record.prepared_checksum,
            record.prepared_bytes,
            record.rgb_checksum,
            record.width,
            record.height,
            record.mode,
        )
        previous = signatures.setdefault(record.prepared_path, signature)
        if previous != signature:
            raise ValueError("Shared prepared paths have inconsistent declarations.")
        source_signature = (
            record.source_bytes,
            record.source_format,
            record.source_mode,
            record.source_width,
            record.source_height,
            record.orientation,
            record.color_policy,
            record.grayscale_converted,
            signature,
        )
        previous_source = source_signatures.setdefault(
            record.source_checksum, source_signature
        )
        if previous_source != source_signature:
            raise ValueError("Identical source bytes have inconsistent declarations.")
    images = directory / "images"
    if images.is_symlink() or not images.is_dir():
        raise ValueError("Prepared image directory is missing or unsafe.")
    found: set[str] = set()
    for index, path in enumerate(images.rglob("*")):
        if index >= 200_000:
            raise ValueError("Too many prepared file entries.")
        if path.is_symlink():
            raise ValueError("Prepared image symlinks are unsupported.")
        if path.is_dir():
            continue
        relative = path.relative_to(directory).as_posix()
        contained_file(directory, relative)
        found.add(relative)
        if len(found) > 200_000:
            raise ValueError("Too many prepared files.")
    if found != set(signatures):
        raise ValueError("Prepared image inventory differs from the frozen records.")


def _validate_coverage(
    manifest: DatasetManifest,
    records: list[DatasetImageRecord],
    all_paths: list[str],
    rejections: list[DatasetRejection],
) -> None:
    """Keep frozen selection and coverage claims consistent with actual records."""
    if manifest.source_kind == "uhd_iqa":
        if (
            manifest.metadata_checksum is None
            or manifest.split_mapping != SPLIT_MAPPING
        ):
            raise ValueError("UHD-IQA provenance or split mapping is missing.")
        if manifest.expected_images is None:
            raise ValueError("UHD-IQA metadata coverage must be declared.")
    elif manifest.split_mapping:
        raise ValueError("Local sources use the frozen generated split policy.")
    if manifest.selected_images != len(all_paths):
        raise ValueError("Selected count disagrees with the source records.")
    if manifest.discovered_images < manifest.selected_images:
        raise ValueError("Discovered count is smaller than the selected input.")
    if manifest.selection is None:
        if manifest.selection_name is not None:
            raise ValueError("Selection name requires frozen source paths.")
        if manifest.discovered_images != manifest.selected_images:
            raise ValueError("Partial imports require a named selection.")
    elif manifest.selection_name is None or manifest.selection != sorted(all_paths):
        raise ValueError("Frozen source selection is inconsistent.")
    for counts, total in (
        (manifest.expected_by_split, manifest.expected_images),
        (manifest.discovered_by_split, manifest.discovered_images),
        (manifest.selected_by_split, manifest.selected_images),
    ):
        if total is not None and sum(counts.values()) > total:
            raise ValueError("Coverage split counts exceed total coverage.")
    actual = Counter(
        [record.upstream_split or record.assigned_split for record in records]
        + [
            record.upstream_split
            for record in rejections
            if record.upstream_split is not None
        ]
    )
    if manifest.source_kind == "uhd_iqa" and dict(actual) != manifest.selected_by_split:
        raise ValueError("Selected source split coverage is inconsistent.")
    if any(
        count > manifest.selected_by_split.get(split or "", 0)
        for split, count in actual.items()
    ):
        raise ValueError("Selected coverage is smaller than accepted records.")
    complete = (
        manifest.selection is None
        and not rejections
        and manifest.discovered_images == len(records)
        and (
            manifest.expected_images is None or manifest.expected_images == len(records)
        )
    )
    if manifest.full_coverage != complete:
        raise ValueError("Full dataset coverage claim is inconsistent.")


def _integrity_failure() -> ApplicationFailure:
    """Keep corrupt record values and filesystem paths out of public errors."""
    return ApplicationFailure(
        "dataset_integrity",
        "Dataset integrity validation failed. Check the revision and prepared files.",
        422,
    )
