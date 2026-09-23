"""Canonical manifests, summary counts, and immutable metadata publication."""

import hashlib
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import PIL
from PIL import features

from backend_service.dataset_groups import assign_groups
from backend_service.dataset_serialization import (
    MAXIMUM_METADATA_BYTES,
    MAXIMUM_RECORD_LINE,
    canonical_json,
    checksum,
)
from schemas.dataset_common import DATASET_SPLITS, SPLIT_MAPPING
from schemas.datasets import (
    DatasetImageRecord,
    DatasetManifest,
    DatasetPreparationRequest,
    DatasetRejection,
    DatasetSummary,
)


def runtime_provenance() -> dict[str, str]:
    """Freeze software versions affecting image preparation and serialization."""
    return {
        "preparation": "stegolab-dataset-v1",
        "pillow": PIL.__version__,
        "libjpeg": features.version("jpg") or "unavailable",
        "zlib": features.version("zlib") or "unavailable",
        "littlecms": features.version("littlecms2") or "unavailable",
    }


def record_counts(
    records: list[DatasetImageRecord], rejections: list[DatasetRejection]
) -> dict[str, object]:
    """Derive every model-input count from validated frozen image records."""
    unique = [
        record
        for record in records
        if record.source_path == record.representative_source_path
    ]
    eligible = [record for record in records if record.eligible]
    unique_eligible = [record for record in unique if record.eligible]
    split_counts = {
        split: sum(record.assigned_split == split for record in records)
        for split in DATASET_SPLITS
    }
    return {
        "accepted_count": len(records),
        "rejection_count": len(rejections),
        "eligible_count": len(eligible),
        "ineligible_count": len(records) - len(eligible),
        "duplicate_count": len(records) - len(unique),
        "unique_eligible_count": len(unique_eligible),
        "split_counts": split_counts,
        "eligible_by_split": {
            split: sum(record.assigned_split == split for record in eligible)
            for split in DATASET_SPLITS
        },
        "unique_eligible_by_split": {
            split: sum(record.assigned_split == split for record in unique_eligible)
            for split in DATASET_SPLITS
        },
        "rejection_reasons": dict(Counter(record.reason for record in rejections)),
        "source_bytes": sum(record.source_bytes for record in records)
        + sum(record.source_bytes for record in rejections),
        "prepared_bytes": sum(
            {record.prepared_path: record.prepared_bytes for record in records}.values()
        ),
    }


def manifest_revision(value: dict[str, object]) -> str:
    """Hash all frozen manifest fields except the revision itself."""
    return checksum(
        canonical_json({key: item for key, item in value.items() if key != "revision"})
    )


def write_manifest(
    directory: Path,
    request: DatasetPreparationRequest,
    records: list[DatasetImageRecord],
    rejections: list[DatasetRejection],
    *,
    before_write: Callable[[int], None],
    metadata_checksum: str | None = None,
    expected_images: int | None = None,
    discovered_images: int | None = None,
    source_provenance: dict[str, str] | None = None,
    expected_by_split: dict[str, int] | None = None,
    discovered_by_split: dict[str, int] | None = None,
    selected_by_split: dict[str, int] | None = None,
) -> DatasetManifest:
    """Write canonical metadata only after grouping and deriving all counts."""
    grouped = assign_groups(records, request.source_kind, request.seed)
    rejected = sorted(rejections, key=lambda record: record.source_path)
    paths = [record.source_path for record in grouped + rejected]
    if len(set(paths)) != len(paths):
        raise ValueError("Accepted and rejected source paths must be unique.")
    records_checksum = _write_records(
        directory / "records.jsonl", grouped, before_write
    )
    rejections_checksum = _write_records(
        directory / "rejections.jsonl", rejected, before_write
    )
    selected_count = len(grouped) + len(rejected)
    discovered_count = (
        selected_count if discovered_images is None else discovered_images
    )
    accepted_coverage = dict(
        Counter(
            [record.upstream_split or record.assigned_split for record in grouped]
            + [
                record.upstream_split
                for record in rejected
                if record.upstream_split is not None
            ]
        )
    )
    data: dict[str, object] = {
        "schema_version": 1,
        "policy_version": request.policy_version,
        "revision": "0" * 64,
        "dataset_name": request.dataset_name,
        "source_kind": request.source_kind,
        "source_url": request.source_url,
        "terms_reference": request.terms_reference,
        "source_provenance": source_provenance or runtime_provenance(),
        "metadata_checksum": metadata_checksum,
        "split_mapping": SPLIT_MAPPING if request.source_kind == "uhd_iqa" else {},
        "seed": request.seed,
        "selection_name": request.selection_name,
        "selection": sorted(request.selection)
        if request.selection is not None
        else None,
        "expected_images": expected_images,
        "discovered_images": discovered_count,
        "selected_images": selected_count,
        "expected_by_split": expected_by_split or {},
        "discovered_by_split": discovered_by_split or accepted_coverage,
        "selected_by_split": selected_by_split or accepted_coverage,
        "full_coverage": request.selection is None
        and not rejected
        and discovered_count == len(grouped)
        and (expected_images is None or expected_images == len(grouped)),
        "records_checksum": records_checksum,
        "rejections_checksum": rejections_checksum,
        "pilot_ready": False,
        **record_counts(grouped, rejected),
    }
    data["revision"] = manifest_revision(data)
    manifest = DatasetManifest.model_validate(data)
    content = canonical_json(manifest.model_dump(mode="json"))
    if len(content) > MAXIMUM_METADATA_BYTES:
        raise ValueError("Dataset manifest exceeds its byte limit.")
    before_write(len(content))
    with (directory / "manifest.json").open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def _write_records[Record: DatasetImageRecord | DatasetRejection](
    path: Path, records: list[Record], before_write: Callable[[int], None]
) -> str:
    """Stream canonical JSONL without buffering the entire record file."""
    digest = hashlib.sha256()
    total = 0
    if len(records) > 200_000:
        raise ValueError("Dataset record count exceeds its limit.")
    with path.open("xb") as stream:
        for record in records:
            line = canonical_json(record.model_dump(mode="json"))
            total += len(line)
            if len(line) > MAXIMUM_RECORD_LINE or total > MAXIMUM_METADATA_BYTES:
                raise ValueError("Dataset record bytes exceed their limit.")
            before_write(len(line))
            stream.write(line)
            digest.update(line)
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest()


def summarize(
    manifest: DatasetManifest, integrity: Literal["not_checked", "verified"]
) -> DatasetSummary:
    """Expose counts with clear warnings about coverage and evaluation readiness."""
    values = manifest.model_dump(mode="json")
    fields = DatasetSummary.model_fields
    summary = {key: value for key, value in values.items() if key in fields}
    warnings = ["Pilot readiness requires a separate near-duplicate and source audit."]
    for split in ("tuning", "held_out"):
        if not manifest.unique_eligible_by_split.get(split, 0):
            warnings.append(
                f"No eligible unique examples in {split}; evaluation is unavailable."
            )
    if not manifest.full_coverage:
        warnings.append("This revision does not cover the complete source dataset.")
    summary.update(integrity=integrity, warnings=warnings)
    return DatasetSummary.model_validate(summary)
