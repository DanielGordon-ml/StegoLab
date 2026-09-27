"""Planned assets, resolved sources and the identities that make fetches reusable."""

import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from backend_service.dataset_serialization import canonical_json, checksum
from backend_service.dataset_sources.source_marker import raw_folder_state
from backend_service.disk_reserve import free_disk_bytes, minimum_free_bytes
from backend_service.failures import ApplicationFailure
from schemas.dataset_common import DatasetSplit, SourceKind
from schemas.dataset_sources import (
    MAXIMUM_IMAGES,
    DatasetInspection,
    DatasetSourceSpec,
    HuggingFaceSourceSpec,
    PlannedAssetRecord,
)

if TYPE_CHECKING:
    from backend_service.dataset_sources.cache import DatasetCache

logger = logging.getLogger(__name__)

MEMBER_POLICY_VERSION = 1
MAXIMUM_LABEL_LENGTH = 64
MAXIMUM_LISTED_ASSETS = 200
ARCHIVE_GROWTH_FACTOR = 1.1
HUGGING_FACE_SELECTION = frozenset(
    "content path_prefix file_names split image_column text_column maximum_files "
    "archive_splits".split()
)
ARCHIVE_SELECTION = frozenset({"archive_splits", "expected_sha256"})
_MESSAGES = {
    "source_split_labels": (
        "Two split folders map to the same label after their names are "
        "simplified. Rename one of them so every label is distinct."
    ),
}


def plan_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for a planning failure code."""
    return ApplicationFailure(code, _MESSAGES[code], 422)


@dataclass(frozen=True)
class PlannedAsset:
    """One file a fetch downloads, imports from an upload, or reuses from cache."""

    url: str | None
    path: str
    expected_size: int | None
    expected_sha256: str | None
    etag: str | None
    resumable: bool
    authorization_host: str | None
    local_file: Path | None = None

    @property
    def basename(self) -> str:
        """Return the last segment of the asset path."""
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ResolvedSource:
    """What an adapter learned about a source before any bytes are fetched."""

    source_kind: SourceKind
    reference: str
    requested_revision: str | None
    resolved_revision: str
    access: Literal["available", "access_required", "not_found", "unsupported"]
    access_guidance: str | None
    content: Literal["images", "text"]
    assets: tuple[PlannedAsset, ...]
    supports_pause: bool
    declared_splits: bool
    split_mapping: dict[str, DatasetSplit]
    member_split_labels: dict[str, str]
    warnings: tuple[str, ...]
    terms_reference: str
    text_column: str | None = None
    image_column: str | None = None


@dataclass(frozen=True)
class MaterializationPlan:
    """A resolved source bound to a raw folder name and its fetch limits."""

    resolved: ResolvedSource
    source_name: str
    identity: str
    maximum_images: int
    training_intended: bool


def asset_identity(
    kind: str, reference: str, revision: str, path: str, expected_sha256: str | None
) -> str:
    """Name one cached file by everything that pins its content."""
    fields = {
        "source_kind": kind,
        "reference": reference,
        "revision": revision,
        "path": path,
        "expected_sha256": expected_sha256,
    }
    return checksum(canonical_json(fields))


def selection_fields(spec: DatasetSourceSpec) -> dict[str, object]:
    """Return the spec fields that choose which files and labels are kept."""
    hub = isinstance(spec, HuggingFaceSourceSpec)
    names = set(HUGGING_FACE_SELECTION if hub else ARCHIVE_SELECTION)
    selection: dict[str, object] = spec.model_dump(mode="json", include=names)
    return dict(sorted(selection.items()))


def materialization_identity(
    spec: DatasetSourceSpec,
    resolved: ResolvedSource,
    *,
    maximum_images: int = MAXIMUM_IMAGES,
) -> str:
    """Name the raw folder content by source, revision, selection and limits.

    The image cap is part of the name because a folder capped at a few images
    holds different content from an uncapped fetch of the same source.
    """
    fields = {
        "source_kind": resolved.source_kind,
        "reference": resolved.reference,
        "revision": resolved.resolved_revision,
        "selection": selection_fields(spec),
        "maximum_images": maximum_images,
        "member_policy_version": MEMBER_POLICY_VERSION,
    }
    return checksum(canonical_json(fields))


def upstream_label(folder_or_archive_name: str) -> str:
    """Turn a folder or archive name into a safe counter key.

    Letters are lower-cased and every other character becomes an underscore.
    A name that does not start with a letter keeps its characters behind the
    prefix ``label_`` so distinct folders stay distinct ("1abc" -> "label_1abc").
    The result never exceeds 64 characters.
    """
    simplified = re.sub(r"[^a-z0-9_]", "_", folder_or_archive_name.lower())
    if not simplified or not simplified[0].isalpha():
        simplified = f"label_{simplified}"
    return simplified[:MAXIMUM_LABEL_LENGTH]


def split_labels(
    spec: DatasetSourceSpec, resolved: ResolvedSource
) -> tuple[dict[str, DatasetSplit], dict[str, str]]:
    """Merge the adapter's split knowledge with the request's folder mapping."""
    split_mapping = dict(resolved.split_mapping)
    member_split_labels = dict(resolved.member_split_labels)
    for folder, assigned in (spec.archive_splits or {}).items():
        label = upstream_label(folder)
        if split_mapping.get(label, assigned) != assigned:
            raise plan_failure("source_split_labels")
        split_mapping[label] = assigned
        member_split_labels[folder] = label
    return split_mapping, member_split_labels


def plan_for(
    spec: DatasetSourceSpec,
    resolved: ResolvedSource,
    *,
    source_name: str,
    maximum_images: int,
    training_intended: bool,
) -> MaterializationPlan:
    """Bind a resolved source to a folder name, applying the request's splits."""
    split_mapping, member_split_labels = split_labels(spec, resolved)
    planned = replace(
        resolved,
        split_mapping=split_mapping,
        member_split_labels=member_split_labels,
        declared_splits=bool(split_mapping),
    )
    return MaterializationPlan(
        resolved=planned,
        source_name=source_name,
        identity=materialization_identity(
            spec, resolved, maximum_images=maximum_images
        ),
        maximum_images=maximum_images,
        training_intended=training_intended,
    )


def suggested_source_name(resolved: ResolvedSource) -> str:
    """Derive a safe folder name from the reference the adapter resolved."""
    reference = resolved.reference.rstrip("/")
    if resolved.source_kind == "hugging_face":
        stem = reference.split("/tree/", 1)[0].rsplit("/", 1)[-1]
    else:
        stem = reference.split(":", 1)[-1].rsplit("/", 1)[-1]
    stem = stem.split(".", 1)[0]
    simplified = re.sub(r"[^a-z0-9_-]", "_", stem.lower()).lstrip("0123456789_-")
    return (simplified or "source")[:MAXIMUM_LABEL_LENGTH]


def _cached_bytes(cache: "DatasetCache", resolved: ResolvedSource) -> int:
    """Sum the sizes of the assets that are already complete in the cache."""
    total = 0
    for asset in resolved.assets:
        identity = asset_identity(
            resolved.source_kind,
            resolved.reference,
            resolved.resolved_revision,
            asset.path,
            asset.expected_sha256,
        )
        entry = cache.find_complete(resolved.source_kind, identity, verify_bytes=False)
        total += 0 if entry is None else entry.manifest.size_bytes
    return total


def inspection_for(
    spec: DatasetSourceSpec,
    resolved: ResolvedSource,
    *,
    source_name: str | None,
    cache: "DatasetCache",
    data_root: Path,
    token_configured: bool,
    maximum_images: int = MAXIMUM_IMAGES,
) -> DatasetInspection:
    """Report what a fetch with the same image cap would download, reuse and need."""
    identity = materialization_identity(spec, resolved, maximum_images=maximum_images)
    name = source_name or suggested_source_name(resolved)
    cached_bytes = _cached_bytes(cache, resolved)
    known = [
        asset.expected_size
        for asset in resolved.assets
        if asset.expected_size is not None
    ]
    all_known = len(known) == len(resolved.assets)
    download_bytes = sum(known) if all_known else None
    materialized_bytes = required_free_bytes = None
    free_bytes = free_disk_bytes(data_root)
    if download_bytes is not None:
        growth = (
            1.0 if resolved.source_kind == "hugging_face" else ARCHIVE_GROWTH_FACTOR
        )
        materialized_bytes = int(download_bytes * growth)
        required_free_bytes = (
            max(download_bytes - cached_bytes, 0)
            + materialized_bytes
            + minimum_free_bytes(data_root)
        )
    listed = resolved.assets[:MAXIMUM_LISTED_ASSETS]
    return DatasetInspection(
        source_kind=resolved.source_kind,
        reference=resolved.reference,
        requested_revision=resolved.requested_revision,
        resolved_revision=resolved.resolved_revision,
        suggested_source_name=name,
        access=resolved.access,
        access_guidance=resolved.access_guidance,
        content=resolved.content,
        asset_count=len(resolved.assets),
        assets=[
            PlannedAssetRecord(
                path=asset.path,
                size_bytes=asset.expected_size,
                sha256=asset.expected_sha256,
            )
            for asset in listed
        ],
        download_bytes=download_bytes,
        materialized_bytes=materialized_bytes,
        cached_bytes=cached_bytes,
        supports_pause=resolved.supports_pause,
        declared_splits=resolved.declared_splits or bool(spec.archive_splits),
        materialization_identity=identity,
        raw_folder=raw_folder_state(data_root, name, identity),
        free_disk_bytes=free_bytes,
        required_free_bytes=required_free_bytes,
        disk_sufficient=(
            None if required_free_bytes is None else free_bytes >= required_free_bytes
        ),
        server_token_configured=token_configured,
        warnings=list(resolved.warnings),
    )
