"""Inspect bounded source headers and reconcile declared image identities."""

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from backend_service.dataset_files import (
    SourceFile,
    dataset_failure,
    hash_source_file,
    read_source_bytes,
)
from backend_service.dataset_image import inspect_dataset_header
from backend_service.dataset_metadata import UhdMetadata, read_uhd_metadata
from backend_service.dataset_scan import SourceInventory, scan_source
from backend_service.failures import ApplicationFailure
from backend_service.image_validation import SourceHeader
from schemas.datasets import DatasetPreparationRequest, DatasetRejection


@dataclass(frozen=True)
class DatasetInventory:
    """Keep lightweight headers, fixed metadata, and selected source entries."""

    source: SourceInventory
    metadata: UhdMetadata | None
    selected: tuple[SourceFile, ...]
    headers: dict[str, SourceHeader]
    rejections: tuple[DatasetRejection, ...]
    expected_by_split: dict[str, int]
    discovered_by_split: dict[str, int]
    source_checksums: dict[str, str]


def inventory_dataset(request: DatasetPreparationRequest) -> DatasetInventory:
    """Check identities and source limits before allocating prepared pixels."""
    root = Path(request.source_directory)
    source = scan_source(root)
    metadata = None
    expected: dict[str, int] = {}
    discovered: dict[str, int] = {}
    candidates = [item for item in source.files if item.kind != "sidecar"]
    if request.source_kind == "uhd_iqa":
        assert request.metadata_file is not None
        metadata = read_uhd_metadata(root, request.metadata_file, source)
        expected = dict(Counter(row.upstream_split for row in metadata.rows.values()))
        image_files = [item for item in candidates if item.kind == "image"]
        names = [Path(item.relative_path).name for item in image_files]
        if len(names) != len(set(names)):
            raise dataset_failure("dataset_metadata")
        discovered = dict(
            Counter(
                metadata.rows[name].upstream_split
                for name in names
                if name in metadata.rows
            )
        )
        if request.selection is None and set(names) != set(metadata.rows):
            raise dataset_failure("dataset_metadata")
    if request.selection is not None:
        chosen = set(request.selection)
        if not chosen.issubset({item.relative_path for item in candidates}):
            raise dataset_failure("dataset_metadata")
        candidates = [item for item in candidates if item.relative_path in chosen]
    if not candidates:
        raise dataset_failure("dataset_empty")
    headers: dict[str, SourceHeader] = {}
    checksums: dict[str, str] = {}
    rejected: list[DatasetRejection] = []
    for item in candidates:
        if metadata is not None and item.kind == "image":
            row = metadata.rows.get(Path(item.relative_path).name)
            if row is None:
                raise dataset_failure("dataset_metadata")
            parts = Path(item.relative_path).parts
            if any(
                part in expected and part != row.upstream_split for part in parts[:-1]
            ):
                raise dataset_failure("dataset_metadata")
        reason = item.rejection_reason
        if reason is None:
            data = read_source_bytes(root, item)
            checksums[item.relative_path] = hashlib.sha256(data).hexdigest()
            try:
                headers[item.relative_path] = inspect_dataset_header(data)
            except ApplicationFailure as failure:
                if failure.code == "image_resources":
                    raise
                reason = failure.code
        if reason is not None:
            if item.relative_path not in checksums:
                checksums[item.relative_path] = hash_source_file(root, item)
            upstream = (
                metadata.rows[Path(item.relative_path).name].upstream_split
                if metadata is not None
                and Path(item.relative_path).name in metadata.rows
                else None
            )
            rejected.append(
                DatasetRejection(
                    source_path=item.relative_path,
                    source_bytes=item.size_bytes,
                    reason=reason,
                    upstream_split=upstream,
                    source_checksum=checksums[item.relative_path],
                )
            )
    return DatasetInventory(
        source,
        metadata,
        tuple(candidates),
        headers,
        tuple(rejected),
        expected,
        discovered,
        checksums,
    )
