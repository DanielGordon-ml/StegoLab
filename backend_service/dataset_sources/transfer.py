"""Bounded, resumable and verified streaming of one remote file to disk."""

import hashlib
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend_service.dataset_sources.transport import (
    SecureTransport,
    TransportResponse,
)
from backend_service.failures import ApplicationFailure

logger = logging.getLogger(__name__)

STREAM_CHUNK_BYTES = 1_048_576
PROGRESS_INTERVAL_BYTES = 4 * 1_048_576
PROGRESS_INTERVAL_SECONDS = 0.5
_MESSAGES = {
    "download_status": "The download server refused the request.",
    "download_too_large": "The download is larger than the allowed size.",
    "download_size_mismatch": "The download ended with an unexpected size.",
    "download_checksum_mismatch": (
        "The downloaded file does not match its expected checksum."
    ),
}


def transfer_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for a transfer failure code."""
    status_code = 400 if code == "download_too_large" else 502
    return ApplicationFailure(code, _MESSAGES[code], status_code)


class DownloadStopped(Exception):
    """The caller asked to stop; the partial file keeps the bytes received."""


@dataclass(frozen=True)
class RemoteAsset:
    """What is known about one remote file before it is fetched."""

    url: str
    basename: str
    expected_size: int | None
    expected_sha256: str | None
    etag: str | None
    resumable: bool


class ProgressSink(Protocol):
    """Receive byte counts while a download runs."""

    def update(self, *, bytes_received: int, bytes_total: int | None) -> None:
        """Record the bytes received so far and the total when it is known."""


@dataclass(frozen=True)
class DownloadOutcome:
    """Size, digest and resume offset of a completed download."""

    size: int
    sha256: str
    resumed_from: int


class PartialFile:
    """Own one ``.part`` file that grows by appends and can be truncated."""

    def __init__(self, path: Path) -> None:
        """Remember the path without touching the file yet."""
        self.path = path
        self._descriptor = -1

    @property
    def size(self) -> int:
        """Return the current file size, or zero when the file is absent."""
        return self.path.stat().st_size if self.path.exists() else 0

    def open_for_append(self) -> int:
        """Open the file for appending (creating it owner-only) and return its size."""
        if self._descriptor < 0:
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
            self._descriptor = os.open(self.path, flags, 0o600)
        return self.size

    def append(self, chunk: bytes) -> None:
        """Write one chunk at the end of the open file."""
        view = memoryview(chunk)
        while view:
            view = view[os.write(self._descriptor, view) :]

    def truncate(self) -> None:
        """Discard every byte so the next append starts from zero."""
        os.ftruncate(self._descriptor, 0)

    def iter_existing(self, limit: int) -> Iterator[bytes]:
        """Read back at most ``limit`` bytes from the start of the file."""
        with self.path.open("rb") as stream:
            for _ in range(0, limit, STREAM_CHUNK_BYTES):
                yield stream.read(min(STREAM_CHUNK_BYTES, limit - stream.tell()))

    def close(self) -> None:
        """Flush appended bytes to storage and release the descriptor."""
        if self._descriptor >= 0:
            try:
                os.fsync(self._descriptor)
            finally:
                os.close(self._descriptor)
                self._descriptor = -1


def download_asset(
    transport: SecureTransport,
    asset: RemoteAsset,
    partial: PartialFile,
    *,
    maximum_bytes: int,
    progress: ProgressSink | None = None,
    stop: Callable[[], bool] = lambda: False,
    stall_timeout: float = 120.0,
) -> DownloadOutcome:
    """Stream the asset into the partial file, resuming when the server allows.

    ``DownloadStopped`` is raised when ``stop`` returns true between chunks;
    the partial file then keeps what arrived. The caller renames the file.
    """
    offset = partial.open_for_append()
    try:
        if offset and not (asset.resumable and asset.etag is not None):
            partial.truncate()
            offset = 0
        byte_range = (offset, None) if offset else None
        if_range = asset.etag if offset else None
        with transport.open(
            "GET", asset.url, byte_range=byte_range, if_range=if_range
        ) as response:
            offset = _accept_response(response, partial, offset)
            digest = hashlib.sha256()
            for existing in partial.iter_existing(offset):
                digest.update(existing)
            length = response.headers.get("content-length", "")
            total = int(length) + offset if length.isdigit() else asset.expected_size
            if total is not None and total > maximum_bytes:
                raise transfer_failure("download_too_large")
            response.set_read_timeout(stall_timeout)
            received, reported_bytes, reported_time = offset, offset, time.monotonic()
            for chunk in response.iter_bytes(STREAM_CHUNK_BYTES):
                received += len(chunk)
                if received > maximum_bytes:
                    raise transfer_failure("download_too_large")
                digest.update(chunk)
                partial.append(chunk)
                now = time.monotonic()
                if progress is not None and (
                    received - reported_bytes >= PROGRESS_INTERVAL_BYTES
                    or now - reported_time >= PROGRESS_INTERVAL_SECONDS
                ):
                    progress.update(bytes_received=received, bytes_total=total)
                    reported_bytes, reported_time = received, now
                if stop():
                    logger.info("download_stopped")
                    raise DownloadStopped()
            if progress is not None and received != reported_bytes:
                progress.update(bytes_received=received, bytes_total=total)
    finally:
        partial.close()
    expected_sizes = {
        value for value in (total, asset.expected_size) if value is not None
    }
    if any(value != received for value in expected_sizes):
        raise transfer_failure("download_size_mismatch")
    checksum = digest.hexdigest()
    if asset.expected_sha256 is not None and checksum != asset.expected_sha256.lower():
        raise transfer_failure("download_checksum_mismatch")
    logger.info("download_completed")
    return DownloadOutcome(size=received, sha256=checksum, resumed_from=offset)


def _accept_response(
    response: TransportResponse, partial: PartialFile, offset: int
) -> int:
    """Return the offset to continue from, restarting when the server sent 200."""
    if response.status == 206 and offset:
        unit, _, span = response.headers.get("content-range", "").partition(" ")
        start = span.split("-", 1)[0]
        if unit == "bytes" and start.isdigit() and int(start) == offset:
            return offset
    elif response.status == 200:
        if offset:
            partial.truncate()
        return 0
    raise transfer_failure("download_status")
