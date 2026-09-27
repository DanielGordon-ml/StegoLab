"""Chunked dataset archive uploads kept as private sessions that expire."""

import logging
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from backend_service.dataset_serialization import canonical_json, checksum
from backend_service.dataset_sources.upload_session_files import (
    UPLOAD_IDENTIFIER,
    StoredUploadSession,
    UploadSessionFiles,
    expected_chunk_bytes,
    is_archive_prefix,
)
from backend_service.disk_reserve import ensure_disk_reserve
from backend_service.failures import ApplicationFailure
from schemas.dataset_uploads import (
    DATASET_UPLOAD_CHUNK_BYTES,
    UPLOAD_RETENTION_SECONDS,
    DatasetUploadChunk,
    DatasetUploadCreateRequest,
    DatasetUploadSession,
)

logger = logging.getLogger(__name__)

MAXIMUM_UPLOAD_STORAGE_BYTES = 6 * 1024**3
UPLOADS_DIRECTORY = "dataset_uploads"
_MESSAGES = {
    "upload_not_found": "No upload session has this identifier. Start a new upload.",
    "upload_expired": (
        "This upload session expired and its parts were removed. Start a new upload."
    ),
    "request_identifier_conflict": (
        "This request identifier was already used for a different upload. "
        "Retry with a new request identifier."
    ),
    "upload_chunk_index": "The part index is outside the part range of this upload.",
    "upload_chunk_size": (
        "The part does not have the exact number of bytes expected at this index."
    ),
    "upload_chunk_mismatch": (
        "A different part was already received at this index. "
        "Discard the upload and start it again."
    ),
    "upload_already_complete": (
        "This upload is already complete and accepts no more parts."
    ),
    "upload_incomplete": "{missing} of {total} parts are still missing.",
    "upload_not_completed": "Complete the upload before its archive can be used.",
    "upload_checksum_mismatch": (
        "The assembled file does not match the expected checksum. "
        "The upload was discarded; start it again."
    ),
    "upload_not_archive": (
        "The uploaded file is not a zip, tar or gzip archive. The upload was discarded."
    ),
    "upload_storage_full": (
        "Upload storage is full. Complete or discard other uploads, then retry."
    ),
    "upload_storage": (
        "Upload data could not be read or written. "
        "Check access and free space, then retry."
    ),
}
_NOT_FOUND_CODES = frozenset({"upload_not_found", "upload_expired"})
_INVALID_CODES = frozenset(
    {
        "upload_chunk_index",
        "upload_chunk_size",
        "upload_checksum_mismatch",
        "upload_not_archive",
    }
)


def upload_failure(code: str, **fields: int) -> ApplicationFailure:
    """Return the fixed plain-language failure for an upload session code."""
    if code in _NOT_FOUND_CODES:
        status_code = 404
    elif code in _INVALID_CODES:
        status_code = 422
    elif code == "upload_storage_full":
        status_code = 507
    elif code == "upload_storage":
        status_code = 503
    else:
        status_code = 409
    return ApplicationFailure(code, _MESSAGES[code].format(**fields), status_code)


def _updated(session: DatasetUploadSession, **changes: object) -> DatasetUploadSession:
    """Return a re-validated copy of a session with some fields changed."""
    return DatasetUploadSession.model_validate({**session.model_dump(), **changes})


class DatasetUploadStore:
    """Keep upload sessions under one state directory for one day each."""

    def __init__(
        self, state_directory: Path, clock: Callable[[], datetime] | None = None
    ) -> None:
        """Remember where sessions live; nothing is created until it is needed."""
        self.root = Path(os.path.abspath(state_directory)) / UPLOADS_DIRECTORY
        self._files = UploadSessionFiles(self.root)
        self._clock = clock or (lambda: datetime.now(UTC))

    def create(self, request: DatasetUploadCreateRequest) -> DatasetUploadSession:
        """Open a session, or return the one a repeated request already opened."""
        with self._locked():
            fingerprint = checksum(canonical_json(request.model_dump(mode="json")))
            existing = self._replayed(request.client_request_identifier)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise upload_failure("request_identifier_conflict")
                return existing.session
            self._check_capacity(request.total_bytes)
            created_at = self._clock()
            session = DatasetUploadSession(
                upload_identifier=f"upload_{uuid4().hex}",
                file_name=request.file_name,
                total_bytes=request.total_bytes,
                chunk_count=math.ceil(request.total_bytes / DATASET_UPLOAD_CHUNK_BYTES),
                created_at=created_at,
                expires_at=created_at + timedelta(seconds=UPLOAD_RETENTION_SECONDS),
            )
            self._files.save_record(
                StoredUploadSession(
                    session=session,
                    client_request_identifier=request.client_request_identifier,
                    fingerprint=fingerprint,
                    expected_sha256=request.expected_sha256,
                )
            )
            self._files.write_request(
                request.client_request_identifier, session.upload_identifier
            )
            logger.info("upload_session_created")
            return session

    def receive_chunk(
        self, identifier: str, index: int, data: bytes
    ) -> DatasetUploadChunk:
        """Store one part of exactly the expected size; identical repeats are fine."""
        with self._locked():
            stored = self._load(identifier)
            session = stored.session
            if session.complete:
                raise upload_failure("upload_already_complete")
            if index < 0 or index >= session.chunk_count:
                raise upload_failure("upload_chunk_index")
            if len(data) != expected_chunk_bytes(session, index):
                raise upload_failure("upload_chunk_size")
            digest = checksum(data)
            if self._files.chunk_size(identifier, index) != len(data):
                self._check_capacity(len(data))
                ensure_disk_reserve(self.root, len(data))
                self._files.write_chunk(identifier, index, data)
            elif self._files.chunk_digest(identifier, index) != digest:
                raise upload_failure("upload_chunk_mismatch")
            if index not in session.received_chunks:
                received = sorted([*session.received_chunks, index])
                session = _updated(session, received_chunks=received)
                self._files.save_record(stored.model_copy(update={"session": session}))
            logger.info("upload_chunk_received")
            return DatasetUploadChunk(
                upload_identifier=identifier,
                index=index,
                bytes=len(data),
                sha256=digest,
                received_chunks=session.received_chunks,
            )

    def read(self, identifier: str) -> DatasetUploadSession:
        """Report a session so a reloaded client can continue where it stopped."""
        with self._locked():
            return self._load(identifier).session

    def complete(self, identifier: str) -> DatasetUploadSession:
        """Join every part into the archive, verify it and free the parts."""
        with self._locked():
            stored = self._load(identifier)
            session = stored.session
            if session.complete:
                self._files.remove_chunks(identifier)
                return session
            present = self._files.verified_parts(identifier, session)
            if len(present) != session.chunk_count:
                session = _updated(session, received_chunks=present)
                self._files.save_record(stored.model_copy(update={"session": session}))
                missing = session.chunk_count - len(present)
                raise upload_failure(
                    "upload_incomplete", missing=missing, total=session.chunk_count
                )
            self._check_capacity(session.chunk_bytes)
            ensure_disk_reserve(self.root, session.chunk_bytes)
            digest, prefix = self._files.assemble(identifier, session.chunk_count)
            if stored.expected_sha256 not in (None, digest):
                self._files.remove_session(identifier, stored.client_request_identifier)
                logger.warning("upload_checksum_mismatch_discarded")
                raise upload_failure("upload_checksum_mismatch")
            if not is_archive_prefix(prefix):
                self._files.remove_session(identifier, stored.client_request_identifier)
                logger.warning("upload_not_archive_discarded")
                raise upload_failure("upload_not_archive")
            session = _updated(session, complete=True, sha256=digest)
            self._files.save_record(stored.model_copy(update={"session": session}))
            self._files.remove_chunks(identifier)
            logger.info("upload_session_completed")
            return session

    def discard(self, identifier: str) -> None:
        """Remove a session and everything it stored; repeating is harmless."""
        with self._locked():
            if UPLOAD_IDENTIFIER.fullmatch(identifier) is None:
                return
            stored = self._files.read_record(identifier)
            request_identifier = stored.client_request_identifier if stored else None
            if self._files.remove_session(identifier, request_identifier):
                logger.info("upload_session_discarded")

    def archive_path(self, identifier: str) -> Path:
        """Return the verified archive of a completed session."""
        with self._locked():
            if not self._load(identifier).session.complete:
                raise upload_failure("upload_not_completed")
            return self._files.archive_path(identifier)

    def sweep(self) -> int:
        """Delete expired and unreadable sessions; return how many were removed."""
        with self._locked():
            removed = 0
            for identifier in self._files.session_identifiers():
                stored = self._files.read_record(identifier)
                request_identifier = (
                    stored.client_request_identifier if stored else None
                )
                if stored is None or self._clock() >= stored.session.expires_at:
                    self._files.remove_session(identifier, request_identifier)
                    removed += 1
            self._files.remove_stale_requests()
            if removed:
                logger.info("upload_sessions_swept")
            return removed

    def used_bytes(self) -> int:
        """Count the bytes that parts and assembled archives occupy."""
        with self._locked():
            return self._files.used_bytes()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialise store changes and turn storage errors into one plain failure."""
        try:
            with self._files.locked():
                yield
        except (OSError, ValueError):
            logger.warning("upload_storage_failure")
            raise upload_failure("upload_storage") from None

    def _load(self, identifier: str) -> StoredUploadSession:
        """Read one live session record, removing it once its lifetime passed."""
        if UPLOAD_IDENTIFIER.fullmatch(identifier) is None:
            raise upload_failure("upload_not_found")
        stored = self._files.read_record(identifier)
        if stored is None:
            raise upload_failure("upload_not_found")
        if self._clock() >= stored.session.expires_at:
            self._files.remove_session(identifier, stored.client_request_identifier)
            logger.info("upload_session_expired_removed")
            raise upload_failure("upload_expired")
        return stored

    def _replayed(self, client_request_identifier: str) -> StoredUploadSession | None:
        """Find the live session an earlier request with this identifier opened."""
        identifier = self._files.read_request(client_request_identifier)
        if identifier is None:
            return None
        try:
            return self._load(identifier)
        except ApplicationFailure as failure:
            if failure.code in _NOT_FOUND_CODES:
                return None
            raise

    def _check_capacity(self, next_bytes: int) -> None:
        """Refuse a write that would push every session past the storage cap."""
        if self._files.used_bytes() + next_bytes > MAXIMUM_UPLOAD_STORAGE_BYTES:
            raise upload_failure("upload_storage_full")
