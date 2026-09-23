"""Reuse matching frozen data before reserving space for another image import."""

import hashlib
import os
import re
import stat
from pathlib import Path

from backend_service.dataset_files import directory_descriptor
from backend_service.dataset_inventory import DatasetInventory
from backend_service.dataset_serialization import canonical_json, read_records
from backend_service.dataset_validation import (
    _integrity_failure,
    load_manifest,
    validate_dataset,
    validated_records,
)
from backend_service.failures import ApplicationFailure
from schemas.datasets import (
    DatasetManifest,
    DatasetPreparationRequest,
    DatasetRejection,
    DatasetSummary,
)

REVISION_NAME = re.compile(r"[a-f0-9]{64}")


def find_reusable_dataset(
    request: DatasetPreparationRequest,
    inventory: DatasetInventory,
    source_provenance: dict[str, str],
) -> DatasetSummary | None:
    """Verify exact inputs before reusing an existing completed dataset revision."""
    directory = Path(request.output_root) / request.dataset_name
    try:
        if not directory.exists() and not directory.is_symlink():
            return None
        candidates = _revision_directories(directory)
        for candidate in candidates:
            manifest = load_manifest(candidate)
            if manifest.revision != candidate.name:
                raise ValueError("Dataset revision and directory disagree.")
            if not _same_request(manifest, request, inventory, source_provenance):
                continue
            _, records = validated_records(candidate)
            rejections = read_records(candidate, "rejections.jsonl", DatasetRejection)
            digest = hashlib.sha256()
            for rejection in rejections:
                digest.update(canonical_json(rejection.model_dump(mode="json")))
            if digest.hexdigest() != manifest.rejections_checksum:
                raise ValueError("Parsed rejection records changed during reuse.")
            stored: dict[str, tuple[int, str]] = {
                record.source_path: (record.source_bytes, record.source_checksum)
                for record in records
            }
            for rejection in rejections:
                checksum = getattr(rejection, "source_checksum", None)
                if not isinstance(checksum, str):
                    break
                stored[rejection.source_path] = (rejection.source_bytes, checksum)
            else:
                current = {
                    item.relative_path: (
                        item.size_bytes,
                        inventory.source_checksums.get(item.relative_path),
                    )
                    for item in inventory.selected
                }
                if stored == current:
                    summary = validate_dataset(candidate)
                    return DatasetSummary.model_validate(
                        summary.model_dump() | {"reused": True}
                    )
        return None
    except (OSError, ValueError, MemoryError, ApplicationFailure):
        raise _integrity_failure() from None


def _revision_directories(directory: Path) -> list[Path]:
    """Bound candidate discovery and refuse symlinks or unrecognized entries."""
    candidates: list[Path] = []
    with directory_descriptor(directory) as descriptor:
        with os.scandir(descriptor) as entries:
            for count, entry in enumerate(entries, start=1):
                if count > 200_000:
                    raise ValueError("Too many dataset revisions.")
                information = entry.stat(follow_symlinks=False)
                if (
                    not stat.S_ISDIR(information.st_mode)
                    or REVISION_NAME.fullmatch(entry.name) is None
                ):
                    raise ValueError("Dataset revision entries are unsafe.")
                candidates.append(directory / entry.name)
    return sorted(candidates)


def _same_request(
    manifest: DatasetManifest,
    request: DatasetPreparationRequest,
    inventory: DatasetInventory,
    provenance: dict[str, str],
) -> bool:
    """Match every frozen setting and current source-coverage declaration."""
    metadata = inventory.metadata
    expected = len(metadata.rows) if metadata is not None else None
    selected = sorted(request.selection) if request.selection is not None else None
    return (
        manifest.dataset_name == request.dataset_name
        and manifest.policy_version == request.policy_version
        and manifest.source_kind == request.source_kind
        and manifest.seed == request.seed
        and manifest.selection_name == request.selection_name
        and manifest.selection == selected
        and manifest.source_url == request.source_url
        and manifest.terms_reference == request.terms_reference
        and manifest.source_provenance == provenance
        and manifest.metadata_checksum == (metadata.sha256 if metadata else None)
        and manifest.expected_images == expected
        and manifest.expected_by_split == inventory.expected_by_split
        and manifest.discovered_images
        == sum(item.kind != "sidecar" for item in inventory.source.files)
        and (
            metadata is None
            or manifest.discovered_by_split == inventory.discovered_by_split
        )
        and manifest.selected_images == len(inventory.selected)
    )
