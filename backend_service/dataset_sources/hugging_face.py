"""Resolve a Hugging Face dataset repository into a list of planned downloads."""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import quote

from backend_service.dataset_sources.hugging_face_files import (
    HUB_HOST,
    ParsedTreePage,
    TreeEntry,
    decode_document,
    hugging_face_failure,
    next_page_url,
    parse_tree_page,
    revision_sha,
    select_files,
    split_assignments,
)
from backend_service.dataset_sources.plans import PlannedAsset, ResolvedSource
from backend_service.hugging_face_credentials import read_hugging_face_token
from schemas.dataset_common import DatasetSplit
from schemas.dataset_sources import HuggingFaceSourceSpec, SourceAccess

if TYPE_CHECKING:
    from backend_service.dataset_sources.upload_sessions import DatasetUploadStore

logger = logging.getLogger(__name__)

API_BASE = f"https://{HUB_HOST}/api/datasets"
RESOLVE_BASE = f"https://{HUB_HOST}/datasets"
MAXIMUM_DOCUMENT_BYTES = 8 * 1024**2
MAXIMUM_TREE_PAGES = 50
RETRY_LIMIT = 3
RETRY_DELAY_SECONDS = 0.5
UNRESOLVED_REVISION = "unresolved"
DENIED_STATUSES = frozenset({401, 403})
GATED_GUIDANCE = (
    "This Hugging Face dataset is gated. Accept its terms on huggingface.co and "
    "set HF_ACCESS_TOKEN (or HF_TOKEN) in the backend environment. Nothing was "
    "downloaded."
)
GATED_WARNING = (
    "This dataset is gated. Downloads only succeed when the configured token "
    "belongs to an account that accepted the dataset terms."
)


class HubResponse(Protocol):
    """The part of a transport response the adapter reads."""

    status: int
    headers: dict[str, str]

    def iter_bytes(self, chunk_size: int = ...) -> Iterator[bytes]:
        """Yield body pieces as they arrive."""

    def close(self) -> None:
        """Release the response."""


class HubTransport(Protocol):
    """The part of a transport the adapter uses: one GET at a time."""

    def open(
        self,
        method: Literal["GET", "HEAD"],
        url: str,
        *,
        headers: dict[str, str] | None = None,
        byte_range: tuple[int, int | None] | None = None,
        if_range: str | None = None,
    ) -> HubResponse:
        """Send one request and return the final response."""


TransportFactory = Callable[[dict[str, str] | None], HubTransport]
SleepFunction = Callable[[float], None]
SplitAssignments = tuple[dict[str, DatasetSplit], dict[str, str]]


@dataclass(frozen=True)
class HubReply:
    """One fully read Hub answer: status, lower-cased headers and body."""

    status: int
    headers: dict[str, str]
    body: bytes


class HubRequester:
    """Send GET requests to the Hub with bounded bodies and short retries."""

    def __init__(self, transport: HubTransport, sleep: SleepFunction) -> None:
        """Keep the transport and the sleep function used between retries."""
        self._transport = transport
        self._sleep = sleep

    def get(self, url: str) -> HubReply:
        """Fetch one address, retrying rate limits and server errors briefly."""
        reply = self._once(url)
        for _retry in range(RETRY_LIMIT):
            if not _retryable(reply.status):
                break
            logger.warning("hugging_face_request_retried")
            self._sleep(RETRY_DELAY_SECONDS)
            reply = self._once(url)
        return reply

    def _once(self, url: str) -> HubReply:
        """Send one request and read its body up to the document limit."""
        response = self._transport.open(
            "GET", url, headers={"Accept": "application/json"}
        )
        try:
            body = _read_bounded(response)
            return HubReply(response.status, dict(response.headers), body)
        finally:
            response.close()


def _retryable(status: int) -> bool:
    """Report whether a status means the Hub may answer on a later attempt."""
    return status == 429 or 500 <= status <= 599


def _read_bounded(response: HubResponse) -> bytes:
    """Collect the body, refusing documents larger than the fixed limit."""
    pieces: list[bytes] = []
    received = 0
    for chunk in response.iter_bytes():
        received += len(chunk)
        if received > MAXIMUM_DOCUMENT_BYTES:
            raise hugging_face_failure("source_listing_invalid")
        pieces.append(chunk)
    return b"".join(pieces)


def _require_success(status: int) -> None:
    """Turn any status outside the success range into a plain failure."""
    if not 200 <= status <= 299:
        logger.warning("hugging_face_unexpected_status")
        raise hugging_face_failure("source_unavailable")


def revision_url(spec: HuggingFaceSourceSpec) -> str:
    """Build the address that resolves the requested revision to a commit."""
    return f"{API_BASE}/{spec.repository}/revision/{quote(spec.revision, safe='')}"


def tree_url(spec: HuggingFaceSourceSpec, sha: str) -> str:
    """Build the recursive listing address, scoped by the prefix when it applies.

    Explicit file names are looked up across the whole tree, so the prefix only
    narrows the listing when no file names are given.
    """
    prefix = None if spec.file_names is not None else spec.path_prefix
    scope = "" if prefix is None else f"/{quote(prefix, safe='/')}"
    return f"{API_BASE}/{spec.repository}/tree/{sha}{scope}?recursive=true"


def _list_tree(
    requester: HubRequester, spec: HuggingFaceSourceSpec, sha: str
) -> ParsedTreePage | None:
    """Read every page of the tree; ``None`` means the Hub denied access."""
    url = tree_url(spec, sha)
    entries: list[TreeEntry] = []
    skipped = 0
    for _page in range(MAXIMUM_TREE_PAGES):
        reply = requester.get(url)
        if reply.status in DENIED_STATUSES:
            logger.warning("hugging_face_access_required")
            return None
        if reply.status == 404:
            raise hugging_face_failure("source_files_empty")
        _require_success(reply.status)
        page = parse_tree_page(decode_document(reply.body))
        entries.extend(page.entries)
        skipped += page.skipped_unsafe
        logger.info("hugging_face_tree_page_read")
        following = next_page_url(reply.headers.get("link"))
        if following is None:
            return ParsedTreePage(tuple(entries), skipped)
        url = following
    raise hugging_face_failure("source_listing_limit")


def _asset(spec: HuggingFaceSourceSpec, sha: str, entry: TreeEntry) -> PlannedAsset:
    """Describe one selected file as a resumable, token-authorized download."""
    return PlannedAsset(
        url=f"{RESOLVE_BASE}/{spec.repository}/resolve/{sha}/"
        f"{quote(entry.path, safe='/')}",
        path=entry.path,
        expected_size=entry.size,
        expected_sha256=entry.sha256,
        etag=None,
        resumable=True,
        authorization_host=HUB_HOST,
    )


def _resolved(
    spec: HuggingFaceSourceSpec,
    revision: str,
    access: SourceAccess,
    *,
    assets: tuple[PlannedAsset, ...] = (),
    warnings: tuple[str, ...] = (),
    splits: SplitAssignments | None = None,
) -> ResolvedSource:
    """Assemble the resolved source; only available sources carry assets."""
    split_mapping, member_split_labels = splits or ({}, {})
    reference_revision = spec.revision if revision == UNRESOLVED_REVISION else revision
    return ResolvedSource(
        source_kind="hugging_face",
        reference=f"{RESOLVE_BASE}/{spec.repository}/tree/{reference_revision}",
        requested_revision=spec.revision,
        resolved_revision=revision,
        access=access,
        access_guidance=GATED_GUIDANCE if access == "access_required" else None,
        content=spec.content,
        assets=assets,
        supports_pause=True,
        declared_splits=bool(split_mapping),
        split_mapping=dict(split_mapping),
        member_split_labels=dict(member_split_labels),
        warnings=warnings,
        terms_reference=spec.terms_reference,
        text_column=spec.text_column,
        image_column=spec.image_column,
    )


def resolve(
    spec: HuggingFaceSourceSpec,
    *,
    transport_factory: TransportFactory,
    uploads: "DatasetUploadStore | None" = None,
    sleep: SleepFunction = time.sleep,
) -> ResolvedSource:
    """Resolve the revision, list the tree and select the files to download.

    The server token, when configured, is attached only for the Hub host. A
    denied revision or listing answers ``access_required`` with fixed guidance
    and stops before any further request; an unknown repository answers
    ``not_found``. Rate limits and server errors are retried a few times.
    """
    token = read_hugging_face_token()
    authorization = None if token is None else {HUB_HOST: f"Bearer {token}"}
    requester = HubRequester(transport_factory(authorization), sleep)
    reply = requester.get(revision_url(spec))
    if reply.status == 404:
        logger.info("hugging_face_repository_not_found")
        return _resolved(spec, UNRESOLVED_REVISION, "not_found")
    if reply.status in DENIED_STATUSES:
        logger.warning("hugging_face_access_required")
        return _resolved(spec, UNRESOLVED_REVISION, "access_required")
    _require_success(reply.status)
    document = decode_document(reply.body)
    sha = revision_sha(document)
    logger.info("hugging_face_revision_resolved")
    gated = isinstance(document, dict) and bool(document.get("gated"))
    if gated and token is None:
        logger.warning("hugging_face_access_required")
        return _resolved(spec, sha, "access_required")
    listing = _list_tree(requester, spec, sha)
    if listing is None:
        return _resolved(spec, sha, "access_required")
    selection = select_files(listing.entries, spec)
    warnings = list(selection.warnings)
    if listing.skipped_unsafe:
        noun = "file" if listing.skipped_unsafe == 1 else "files"
        warnings.append(f"Skipped {listing.skipped_unsafe} {noun} with unsafe names.")
    if gated:
        warnings.append(GATED_WARNING)
    return _resolved(
        spec,
        sha,
        "available",
        assets=tuple(_asset(spec, sha, entry) for entry in selection.entries),
        warnings=tuple(warnings),
        splits=split_assignments(spec, selection.entries),
    )
