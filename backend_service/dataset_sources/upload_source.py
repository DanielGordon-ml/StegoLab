"""Resolve a completed browser upload into a source that a fetch can import."""

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import replace

from backend_service.dataset_sources.plans import (
    PlannedAsset,
    ResolvedSource,
    split_labels,
)
from backend_service.dataset_sources.transport import SecureTransport
from backend_service.dataset_sources.upload_sessions import (
    DatasetUploadStore,
    upload_failure,
)
from backend_service.failures import ApplicationFailure
from schemas.dataset_common import relative_path
from schemas.dataset_sources import DatasetSourceSpec, UploadSourceSpec
from schemas.dataset_uploads import DatasetUploadSession

logger = logging.getLogger(__name__)

TransportFactory = Callable[[dict[str, str] | None], SecureTransport]
REFERENCE_PREFIX = "upload:"
FALLBACK_BASENAME = "archive"
MAXIMUM_BASENAME_LENGTH = 255
RENAMED_WARNING = "The uploaded file name was simplified to a safe name for storage."
_PATH_SEPARATORS = re.compile(r"[/\\]")
_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")
_FAILURES: dict[str, tuple[str, int]] = {
    "upload_store_unavailable": (
        "Uploaded archives cannot be used here because no upload storage is "
        "configured. Choose a Hugging Face or https archive source instead.",
        503,
    ),
    "upload_source_kind": (
        "This source is not an uploaded archive. Use the adapter that matches "
        "its source kind.",
        422,
    ),
    "upload_incomplete": (
        "Every part arrived but the upload was not completed. Complete the "
        "upload, then retry.",
        409,
    ),
}


def upload_source_failure(code: str) -> ApplicationFailure:
    """Return the fixed plain-language failure for an upload source code."""
    message, status_code = _FAILURES[code]
    return ApplicationFailure(code, message, status_code)


def safe_basename(file_name: str) -> str:
    """Turn any uploaded file name into one safe relative path segment.

    Only the last path segment counts. Accented letters lose their accents,
    every run of other characters outside letters, digits, dot, underscore
    and dash becomes one underscore, and leading dots go so the file is
    never treated as a hidden sidecar. An empty result falls back to a
    fixed name, and the result is checked with the shared path rules.
    """
    segment = _PATH_SEPARATORS.split(file_name)[-1].strip()
    letters = unicodedata.normalize("NFKD", segment)
    unaccented = "".join(
        character for character in letters if not unicodedata.combining(character)
    )
    simplified = _UNSAFE_CHARACTERS.sub("_", unaccented).lstrip(".")
    return relative_path(simplified[:MAXIMUM_BASENAME_LENGTH] or FALLBACK_BASENAME)


def _completed_digest(session: DatasetUploadSession) -> str:
    """Return the archive digest of a completed session; refuse any other."""
    if session.complete and session.sha256 is not None:
        return session.sha256
    missing = session.chunk_count - len(session.received_chunks)
    if missing:
        raise upload_failure(
            "upload_incomplete", missing=missing, total=session.chunk_count
        )
    raise upload_source_failure("upload_incomplete")


def resolve(
    spec: DatasetSourceSpec,
    *,
    transport_factory: TransportFactory,
    uploads: DatasetUploadStore | None,
) -> ResolvedSource:
    """Describe a completed upload as one local asset that needs no download.

    The transport factory belongs to the shared adapter signature and is never
    called, because the archive already sits in the upload store.
    """
    if not isinstance(spec, UploadSourceSpec):
        raise upload_source_failure("upload_source_kind")
    if uploads is None:
        raise upload_source_failure("upload_store_unavailable")
    session = uploads.read(spec.upload_identifier)
    digest = _completed_digest(session)
    basename = safe_basename(session.file_name)
    asset = PlannedAsset(
        url=None,
        path=basename,
        expected_size=session.total_bytes,
        expected_sha256=digest,
        etag=None,
        resumable=False,
        authorization_host=None,
        local_file=uploads.archive_path(spec.upload_identifier),
    )
    warnings = () if basename == session.file_name else (RENAMED_WARNING,)
    unlabelled = ResolvedSource(
        source_kind="upload",
        reference=f"{REFERENCE_PREFIX}{session.file_name}",
        requested_revision=None,
        resolved_revision=digest,
        access="available",
        access_guidance=None,
        content="images",
        assets=(asset,),
        supports_pause=False,
        declared_splits=False,
        split_mapping={},
        member_split_labels={},
        warnings=warnings,
        terms_reference=spec.terms_reference,
    )
    split_mapping, member_split_labels = split_labels(spec, unlabelled)
    logger.info("upload_source_resolved")
    return replace(
        unlabelled,
        split_mapping=split_mapping,
        member_split_labels=member_split_labels,
        declared_splits=bool(split_mapping),
    )
