"""Place audit reports beside the dataset folders and write them without follow."""

import os
from pathlib import Path

from backend_service.dataset_files import directory_descriptor
from backend_service.dataset_serialization import canonical_json
from backend_service.failures import ApplicationFailure
from backend_service.near_duplicate_hash import audit_failure
from schemas.near_duplicate_audit import (
    NearDuplicateAuditReport,
    NearDuplicateAuditRequest,
)

AUDITS_FOLDER = ".audits"
REPORT_NAME = "near_duplicate_audit.json"


def audit_output_path(
    output_root: Path, dataset_name: str, audit_identifier: str
) -> Path:
    """Place reports under the output root, never inside a revision."""
    return output_root / AUDITS_FOLDER / dataset_name / audit_identifier / REPORT_NAME


def _dataset_name_folders(request: NearDuplicateAuditRequest) -> list[Path]:
    """List the dataset name folders that hold the revisions the audit reads."""
    directories = [request.dataset_directory, request.benchmark_dataset_directory]
    return [
        Path(os.path.abspath(directory)).parent
        for directory in directories
        if directory is not None
    ]


def resolve_output_root(request: NearDuplicateAuditRequest) -> Path:
    """Choose the report root and refuse one inside a dataset name folder.

    The default is the folder that holds the dataset name folder. A root equal
    to or inside a revision or its dataset name folder would spoil the immutable
    revision tree or the revision listing, so it is refused before any work.
    """
    folders = _dataset_name_folders(request)
    root = (
        folders[0].parent
        if request.output_root is None
        else Path(os.path.abspath(request.output_root))
    )
    if any(root == folder or folder in root.parents for folder in folders):
        raise audit_failure("audit_output_overlaps")
    return root


def _write_exclusive(folder: int, content: bytes) -> None:
    """Create the report inside the opened folder and flush it to disk."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(REPORT_NAME, flags, 0o600, dir_fd=folder)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def write_report(root: Path, report: NearDuplicateAuditReport) -> Path:
    """Create the report folder chain without following symbolic links.

    Every parent is opened with no-follow descriptors, the identifier folder
    must not exist yet, and the report is created exclusively.
    """
    path = audit_output_path(root, report.dataset_name, report.audit_identifier)
    content = canonical_json(report.model_dump(mode="json"))
    try:
        with directory_descriptor(path.parent.parent, create=True) as parent:
            try:
                os.mkdir(path.parent.name, mode=0o700, dir_fd=parent)
            except FileExistsError:
                raise audit_failure("audit_output_exists") from None
            folder = os.open(
                path.parent.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            try:
                _write_exclusive(folder, content)
            finally:
                os.close(folder)
    except ApplicationFailure as failure:
        if failure.code.startswith("audit_"):
            raise
        raise audit_failure("audit_storage") from None
    except OSError:
        raise audit_failure("audit_storage") from None
    return path
