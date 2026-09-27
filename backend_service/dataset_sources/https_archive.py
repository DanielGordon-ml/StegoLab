"""Describe one archive behind a plain https address before any bytes move."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit, urlunsplit

from backend_service.dataset_sources.plans import (
    PlannedAsset,
    ResolvedSource,
    split_labels,
)
from backend_service.dataset_sources.transport import (
    SecureTransport,
    TransportResponse,
)
from backend_service.failures import ApplicationFailure
from schemas.dataset_common import relative_path
from schemas.dataset_sources import (
    DatasetSourceSpec,
    HttpsArchiveSourceSpec,
    SourceAccess,
)

if TYPE_CHECKING:
    from backend_service.dataset_sources.upload_sessions import DatasetUploadStore

logger = logging.getLogger(__name__)

TransportFactory = Callable[[dict[str, str] | None], SecureTransport]
DEFAULT_BASENAME = "archive.zip"
UNVERSIONED = "unversioned"
MAXIMUM_BASENAME_BYTES = 255
MAXIMUM_REVISION_LENGTH = 128
HEAD_FALLBACK_STATUSES = frozenset({405, 501})
NOT_FOUND_STATUSES = frozenset({404, 410})
ACCESS_REQUIRED_STATUSES = frozenset({401, 403})
ACCESS_GUIDANCE = {
    "not_found": (
        "The archive server reports nothing at this address. Check the address "
        "with the dataset publisher. Nothing was downloaded."
    ),
    "access_required": (
        "The archive server refused this address. Check that the archive is "
        "publicly downloadable, or accept the publisher's terms and use the "
        "address they provide. Nothing was downloaded."
    ),
}
RESUME_WARNING = (
    "The server does not support resuming this archive. A stopped fetch will "
    "start again from the beginning."
)
SIZE_WARNING = (
    "The server did not report the archive size, so disk needs cannot be "
    "estimated before the download."
)
_MESSAGES = {
    "source_kind_mismatch": (
        "This source is not an https archive. Use the adapter for its kind."
    ),
    "source_unavailable": (
        "The archive server did not answer as expected. Try again later or "
        "check the address with the publisher."
    ),
    "source_too_large": (
        "The archive is larger than the allowed download size. Raise "
        "maximum_download_bytes or choose a smaller archive."
    ),
}


def archive_source_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for an archive source failure code."""
    status_code = {"source_kind_mismatch": 422, "source_too_large": 400}.get(code, 502)
    return ApplicationFailure(code, _MESSAGES[code], status_code)


@dataclass(frozen=True)
class ArchiveHeaders:
    """What the server said about the archive without sending its body."""

    status: int
    size: int | None
    etag: str | None
    last_modified: str | None
    accepts_ranges: bool


def headers_from(response: TransportResponse) -> ArchiveHeaders:
    """Read the size, version and range support out of one response.

    Weak ETags (``W/...``) cannot pin bytes for a resumed download, so they are
    ignored like a missing ETag; over-long values are ignored the same way.
    """
    headers = response.headers
    length = headers.get("content-length", "").strip()
    try:
        size: int | None = int(length) if length.isdigit() else None
    except ValueError:
        size = None
    etag = headers.get("etag", "").strip()
    if etag.upper().startswith("W/") or len(etag) > MAXIMUM_REVISION_LENGTH:
        etag = ""
    return ArchiveHeaders(
        status=response.status,
        size=size,
        etag=etag or None,
        last_modified=headers.get("last-modified", "").strip() or None,
        accepts_ranges=headers.get("accept-ranges", "").strip().lower() == "bytes",
    )


def probe_archive(transport: SecureTransport, url: str) -> ArchiveHeaders:
    """Ask for headers with HEAD, or with a GET that is closed before its body."""
    with transport.open("HEAD", url) as response:
        if response.status not in HEAD_FALLBACK_STATUSES:
            return headers_from(response)
    logger.info("archive_head_refused_trying_get")
    with transport.open("GET", url) as response:
        return headers_from(response)


def archive_basename(url: str) -> str:
    """Name the archive after the last segment of its address path.

    Empty, hidden, over-long or unsafe segments fall back to ``archive.zip``.
    """
    segment = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    if (
        "/" in segment
        or segment.startswith(".")
        or len(segment.encode()) > MAXIMUM_BASENAME_BYTES
    ):
        return DEFAULT_BASENAME
    try:
        relative_path(segment)
    except ValueError:
        return DEFAULT_BASENAME
    return segment


def archive_reference(url: str) -> str:
    """Return the address without its query so the same file has one name."""
    return urlunsplit(urlsplit(url)._replace(query="", fragment=""))


def revision_for(headers: ArchiveHeaders) -> str:
    """Pin the content by ETag, else by last change and size, else not at all."""
    if headers.etag is not None:
        return headers.etag
    if headers.last_modified is not None and headers.size is not None:
        revision = f"{headers.last_modified}+{headers.size}"
        if len(revision) <= MAXIMUM_REVISION_LENGTH:
            return revision
    return UNVERSIONED


def access_for(status: int) -> SourceAccess:
    """Map the response status to an access state, refusing surprises."""
    if 200 <= status < 300:
        return "available"
    if status in NOT_FOUND_STATUSES:
        return "not_found"
    if status in ACCESS_REQUIRED_STATUSES:
        return "access_required"
    raise archive_source_failure("source_unavailable")


def resolve(
    spec: DatasetSourceSpec,
    *,
    transport_factory: TransportFactory,
    uploads: "DatasetUploadStore | None",
) -> ResolvedSource:
    """Describe the archive from its headers alone and plan one download.

    The upload store belongs to the shared adapter signature and is unused,
    because the archive lives on a remote server. No authorization is attached:
    the factory is called with ``None``.
    """
    if not isinstance(spec, HttpsArchiveSourceSpec):
        raise archive_source_failure("source_kind_mismatch")
    logger.info("archive_resolve_started")
    headers = probe_archive(transport_factory(None), spec.url)
    access = access_for(headers.status)
    if access != "available":
        logger.warning("archive_not_available")
        headers = ArchiveHeaders(headers.status, None, None, None, False)
    elif headers.size is not None and headers.size > spec.maximum_download_bytes:
        logger.warning("archive_larger_than_allowed")
        raise archive_source_failure("source_too_large")
    supports_pause = headers.accepts_ranges and headers.etag is not None
    asset = PlannedAsset(
        url=spec.url,
        path=archive_basename(spec.url),
        expected_size=headers.size,
        expected_sha256=spec.expected_sha256,
        etag=headers.etag,
        resumable=supports_pause,
        authorization_host=None,
    )
    notes: list[str] = []
    if access == "available" and not supports_pause:
        notes.append(RESUME_WARNING)
    if access == "available" and headers.size is None:
        notes.append(SIZE_WARNING)
    resolved = ResolvedSource(
        source_kind="https_archive",
        reference=archive_reference(spec.url),
        requested_revision=None,
        resolved_revision=revision_for(headers),
        access=access,
        access_guidance=ACCESS_GUIDANCE.get(access),
        content="images",
        assets=(asset,),
        supports_pause=supports_pause,
        declared_splits=bool(spec.archive_splits),
        split_mapping={},
        member_split_labels={},
        warnings=tuple(notes),
        terms_reference=spec.terms_reference,
    )
    split_mapping, member_split_labels = split_labels(spec, resolved)
    logger.info("archive_resolved")
    return replace(
        resolved, split_mapping=split_mapping, member_split_labels=member_split_labels
    )
