"""Provenance markers stored beside materialized raw source files."""

import logging
import os
import stat
from pathlib import Path
from typing import Literal

from backend_service.dataset_files import directory_descriptor, relative_parts
from backend_service.dataset_serialization import canonical_json, checksum
from backend_service.dataset_storage import DatasetWriter
from backend_service.failures import ApplicationFailure
from schemas.dataset_common import REMOTE_SOURCE_KINDS
from schemas.dataset_records import DatasetPreparationRequest
from schemas.dataset_sources import SourceMarker

logger = logging.getLogger(__name__)

MARKER_NAME = ".stegolab_source.json"
METADATA_NAME = "source-metadata.csv"
MAXIMUM_MARKER_BYTES = 4 * 1024**2
RawFolderState = Literal["available", "reusable", "conflict"]
_MESSAGES = {
    "source_marker_invalid": (
        "The provenance marker of this source folder could not be read. "
        "Fetch the source again or choose another folder."
    ),
    "source_marker_missing": (
        "This folder has no completed provenance marker for its source kind. "
        "Fetch the source again before preparing it."
    ),
}


def source_marker_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for a source marker failure code."""
    return ApplicationFailure(code, _MESSAGES[code], 422)


def read_source_marker(directory: Path) -> SourceMarker | None:
    """Read the bounded marker of one folder; a missing marker is not an error."""
    try:
        information = os.lstat(directory / MARKER_NAME)
    except FileNotFoundError:
        return None
    except OSError:
        raise source_marker_failure("source_marker_invalid") from None
    try:
        if (
            not stat.S_ISREG(information.st_mode)
            or information.st_size > MAXIMUM_MARKER_BYTES
        ):
            raise ValueError("The source marker is not a small regular file.")
        with directory_descriptor(directory) as parent:
            descriptor = os.open(
                MARKER_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            with os.fdopen(descriptor, "rb") as stream:
                data = stream.read(MAXIMUM_MARKER_BYTES + 1)
        if len(data) != information.st_size:
            raise ValueError("The source marker changed while it was read.")
        return SourceMarker.model_validate_json(data)
    except (OSError, ValueError, ApplicationFailure):
        logger.warning("source_marker_unreadable")
        raise source_marker_failure("source_marker_invalid") from None


def write_source_marker(writer: DatasetWriter, marker: SourceMarker) -> None:
    """Store the marker as canonical JSON inside the writer's stage."""
    writer.write_bytes(MARKER_NAME, canonical_json(marker.model_dump(mode="json")))
    logger.info("source_marker_written")


def raw_folder_state(
    data_root: Path, source_name: str, identity: str
) -> RawFolderState:
    """Say whether a source folder is free, reusable for this identity, or taken."""
    try:
        if len(relative_parts(source_name)) != 1:
            return "conflict"
    except ApplicationFailure:
        return "conflict"
    folder = data_root / source_name
    try:
        information = os.lstat(folder)
    except FileNotFoundError:
        return "available"
    except OSError:
        return "conflict"
    if not stat.S_ISDIR(information.st_mode):
        return "conflict"
    try:
        marker = read_source_marker(folder)
    except ApplicationFailure:
        return "conflict"
    if (
        marker is not None
        and marker.completed
        and marker.materialization_identity == identity
    ):
        return "reusable"
    return "conflict"


def marker_provenance(request: DatasetPreparationRequest) -> dict[str, str]:
    """Freeze the fetched source's identity for remote kinds; nothing for others."""
    if request.source_kind not in REMOTE_SOURCE_KINDS:
        return {}
    marker = read_source_marker(Path(request.source_directory))
    if (
        marker is None
        or not marker.completed
        or marker.source_kind != request.source_kind
    ):
        raise source_marker_failure("source_marker_missing")
    assets = [asset.model_dump(mode="json") for asset in marker.assets]
    members = {
        "member_count": marker.member_count,
        "rejected_member_count": marker.rejected_member_count,
    }
    return {
        "source_name": marker.source_name,
        "source_reference": marker.reference,
        "source_revision": marker.resolved_revision,
        "source_materialization": marker.materialization_identity,
        "source_assets_checksum": checksum(canonical_json(assets)),
        "source_members_checksum": checksum(canonical_json(members)),
        "training_intended": str(request.training_intended).lower(),
    }


def preparation_document(
    source: Path, dataset_name: str, output_root: Path
) -> dict[str, object]:
    """Build the preparation request for a folder from its marker, else local."""
    marker = read_source_marker(source)
    if marker is None or not marker.completed:
        return {
            "schema_version": 1,
            "source_kind": "local",
            "metadata_file": None,
            "source_directory": str(source),
            "output_root": str(output_root),
            "dataset_name": dataset_name,
            "seed": 0,
        }
    return {
        "schema_version": 1,
        "source_kind": marker.source_kind,
        "metadata_file": METADATA_NAME if marker.declared_splits else None,
        "source_directory": str(source),
        "output_root": str(output_root),
        "dataset_name": dataset_name,
        "seed": 0,
        "source_url": marker.source_url,
        "terms_reference": marker.terms_reference,
        "split_mapping": dict(marker.split_mapping),
        "training_intended": True,
    }
