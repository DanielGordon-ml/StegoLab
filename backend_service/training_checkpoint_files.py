"""Atomic checkpoint writes, bounded reads, locking, and disk headroom."""

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

from backend_service.failures import ApplicationFailure

MAXIMUM_STATE_BYTES = 1024**3


def checkpoint_failure() -> ApplicationFailure:
    """Hide file contents and low-level errors behind actionable safe guidance."""
    return ApplicationFailure(
        "checkpoint_unavailable",
        "The checkpoint could not be verified, saved, or restored. Check its "
        "compatibility, storage access, and free space. Previous saved files are kept.",
    )


def canonical_bytes(value: object) -> bytes:
    """Serialize metadata and frozen settings with stable strict JSON rules."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def checksum(path: Path) -> str:
    """Hash a regular bounded state file without holding another full copy."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular checkpoint file is required.")
    if path.stat().st_size > MAXIMUM_STATE_BYTES:
        raise ValueError("The checkpoint file exceeds its limit.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_space(directory: Path, additional_bytes: int) -> None:
    """Keep at least ten GiB or ten percent free after every pending write."""
    usage = shutil.disk_usage(directory)
    reserve = max(10 * 1024**3, (usage.total + 9) // 10)
    if usage.free - additional_bytes < reserve:
        raise checkpoint_failure()


def sync_directory(directory: Path) -> None:
    """Make a completed atomic rename durable on the containing filesystem."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: object) -> None:
    """Publish JSON only after its temporary file has been fully flushed."""
    data = canonical_bytes(value)
    require_space(path.parent, len(data))
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.pending"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> bytes:
    """Read bounded regular metadata without following an explicit symlink."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024**2:
        raise ValueError("Checkpoint metadata is missing or too large.")
    return path.read_bytes()


@contextmanager
def checkpoint_lock(directory: Path) -> Iterator[None]:
    """Keep concurrent checkpoint writers from racing on publication or pruning."""
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise checkpoint_failure()
    descriptor = os.open(
        directory / ".checkpoint.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise checkpoint_failure() from None
        yield
    finally:
        os.close(descriptor)


class ReservedWriter:
    """Check the disk reserve before each chunk written by Torch serialization."""

    def __init__(self, stream: BinaryIO, directory: Path) -> None:
        """Keep a caller-owned stream and cumulative state-file byte count."""
        self.stream = stream
        self.directory = directory
        self.bytes_written = 0

    def write(self, data: Any) -> int:
        """Bound the complete checkpoint and preserve space before writing bytes."""
        size = len(data)
        if self.bytes_written + size > MAXIMUM_STATE_BYTES:
            raise checkpoint_failure()
        require_space(self.directory, size)
        written = self.stream.write(data)
        self.bytes_written += written
        return written

    def flush(self) -> None:
        """Flush through the file-like interface expected by Torch."""
        self.stream.flush()
