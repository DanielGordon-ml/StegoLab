"""On-disk layout and owner-only file operations for upload sessions."""

import fcntl
import hashlib
import os
import re
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from backend_service.dataset_serialization import canonical_json
from schemas.base import StrictRecord
from schemas.configuration import RequestIdentifier
from schemas.dataset_common import SHA256
from schemas.dataset_uploads import DatasetUploadSession

COPY_BLOCK_BYTES = 1024**2
ARCHIVE_PREFIX_BYTES = 512
MAXIMUM_RECORD_BYTES = 64 * 1024
OWNER_ONLY_FILE = 0o600
OWNER_ONLY_DIRECTORY = 0o700
UPLOAD_IDENTIFIER = re.compile(r"^upload_[0-9a-f]{32}$")
REQUESTS_DIRECTORY = "requests"
CHUNKS_DIRECTORY = "chunks"
SESSION_RECORD_NAME = "session.json"
ARCHIVE_NAME = "archive.bin"
LOCK_NAME = ".uploads.lock"
_OPEN_NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
_OPEN_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK


class StoredUploadSession(StrictRecord):
    """The session record kept on disk, with fields clients never receive."""

    session: DatasetUploadSession
    client_request_identifier: RequestIdentifier
    fingerprint: SHA256
    expected_sha256: SHA256 | None


def expected_chunk_bytes(session: DatasetUploadSession, index: int) -> int:
    """Return the exact size the part with this index must have."""
    if index == session.chunk_count - 1:
        return session.total_bytes - index * session.chunk_bytes
    return session.chunk_bytes


def is_archive_prefix(prefix: bytes) -> bool:
    """Recognise zip, gzip and tar files by their leading bytes."""
    return (
        prefix.startswith(b"PK\x03\x04")
        or prefix.startswith(b"\x1f\x8b")
        or (len(prefix) >= 262 and prefix[257:262] == b"ustar")
    )


def _create_directory(path: Path) -> None:
    """Create each missing level of a directory tree as owner-only."""
    if not path.exists():
        _create_directory(path.parent)
        path.mkdir(mode=OWNER_ONLY_DIRECTORY, exist_ok=True)


def _fsync_directory(path: Path) -> None:
    """Flush a directory's entries so new files and renames survive a crash."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, data: bytes) -> None:
    """Create an owner-only file exclusively and flush it to storage."""
    descriptor = os.open(path, _OPEN_NEW, OWNER_ONLY_FILE)
    with os.fdopen(descriptor, "wb") as output:
        output.write(data)
        os.fsync(output.fileno())
    _fsync_directory(path.parent)


def _replace_file(path: Path, data: bytes) -> None:
    """Rewrite a small file atomically through an owner-only sibling file."""
    pending = path.with_name(path.name + ".pending")
    pending.unlink(missing_ok=True)
    _write_new_file(pending, data)
    os.replace(pending, path)
    _fsync_directory(path.parent)


def _read_file(path: Path, maximum: int) -> bytes | None:
    """Read a bounded regular file without following links; None when absent."""
    try:
        descriptor = os.open(path, _OPEN_READ)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        information = os.fstat(stream.fileno())
        if not stat.S_ISREG(information.st_mode) or information.st_size > maximum:
            raise ValueError("The upload record is not a regular file within limits.")
        data = stream.read(maximum + 1)
    if len(data) != information.st_size:
        raise ValueError("The upload record changed while it was being read.")
    return data


def _regular_file_size(path: Path) -> int | None:
    """Return a regular file's size, or None when it is absent or not regular."""
    try:
        information = path.lstat()
    except FileNotFoundError:
        return None
    return information.st_size if stat.S_ISREG(information.st_mode) else None


def _iter_file_blocks(path: Path) -> Iterator[bytes]:
    """Yield a regular file's bytes in bounded blocks without following links."""
    descriptor = os.open(path, _OPEN_READ)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("An upload part is not a regular file.")
        while block := stream.read(COPY_BLOCK_BYTES):
            yield block


def _remove_entry(path: Path) -> bool:
    """Delete a file, link or directory tree this store owns; False when absent."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        return False
    _fsync_directory(path.parent)
    return True


class UploadSessionFiles:
    """Read and write one upload store's directory tree as owner-only files."""

    def __init__(self, root: Path) -> None:
        """Remember the store root; nothing is created until it is locked."""
        self.root = root

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Create the store tree and hold its exclusive advisory lock."""
        _create_directory(self.root / REQUESTS_DIRECTORY)
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
        descriptor = os.open(self.root / LOCK_NAME, flags, OWNER_ONLY_FILE)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def session_directory(self, identifier: str) -> Path:
        """Return the directory that holds one session."""
        return self.root / identifier

    def archive_path(self, identifier: str) -> Path:
        """Return the assembled archive file of one session."""
        return self.session_directory(identifier) / ARCHIVE_NAME

    def chunk_path(self, identifier: str, index: int) -> Path:
        """Return the file that holds one received part."""
        chunks = self.session_directory(identifier) / CHUNKS_DIRECTORY
        return chunks / f"{index:06d}.part"

    def request_path(self, client_request_identifier: str) -> Path:
        """Return the file that maps one client request to its session."""
        return self.root / REQUESTS_DIRECTORY / client_request_identifier

    def session_identifiers(self) -> list[str]:
        """List the session directories in the store, oldest name first."""
        return sorted(
            entry.name
            for entry in self.root.iterdir()
            if not entry.is_symlink() and UPLOAD_IDENTIFIER.fullmatch(entry.name)
        )

    def read_record(self, identifier: str) -> StoredUploadSession | None:
        """Parse a session record, or None when it is absent or unreadable."""
        path = self.session_directory(identifier) / SESSION_RECORD_NAME
        data = _read_file(path, MAXIMUM_RECORD_BYTES)
        if data is None:
            return None
        try:
            return StoredUploadSession.model_validate_json(data)
        except ValidationError:
            return None

    def save_record(self, stored: StoredUploadSession) -> None:
        """Rewrite a session record atomically inside its own directory."""
        directory = self.session_directory(stored.session.upload_identifier)
        _create_directory(directory)
        data = canonical_json(stored.model_dump(mode="json"))
        _replace_file(directory / SESSION_RECORD_NAME, data)

    def read_request(self, client_request_identifier: str) -> str | None:
        """Return the session identifier a client request file points at."""
        data = _read_file(self.request_path(client_request_identifier), 64)
        return None if data is None else data.decode("ascii", "replace").strip()

    def write_request(self, client_request_identifier: str, identifier: str) -> None:
        """Point a client request file at its session for later replays."""
        path = self.request_path(client_request_identifier)
        _replace_file(path, identifier.encode("ascii"))

    def remove_session(self, identifier: str, request_identifier: str | None) -> bool:
        """Delete a session directory and the request file that points at it."""
        removed = _remove_entry(self.session_directory(identifier))
        if request_identifier and self.read_request(request_identifier) == identifier:
            _remove_entry(self.request_path(request_identifier))
        return removed

    def remove_stale_requests(self) -> None:
        """Delete request files whose session directory no longer exists."""
        for path in sorted((self.root / REQUESTS_DIRECTORY).iterdir()):
            target = self.read_request(path.name) or ""
            if not UPLOAD_IDENTIFIER.fullmatch(target) or not (
                self.session_directory(target).is_dir()
            ):
                _remove_entry(path)

    def chunk_size(self, identifier: str, index: int) -> int | None:
        """Return the stored size of one part, or None when it is absent."""
        return _regular_file_size(self.chunk_path(identifier, index))

    def chunk_digest(self, identifier: str, index: int) -> str:
        """Hash one stored part in blocks without loading it whole."""
        digest = hashlib.sha256()
        for block in _iter_file_blocks(self.chunk_path(identifier, index)):
            digest.update(block)
        return digest.hexdigest()

    def write_chunk(self, identifier: str, index: int, data: bytes) -> None:
        """Store one new part exclusively, replacing a short leftover if any."""
        path = self.chunk_path(identifier, index)
        _create_directory(path.parent)
        path.unlink(missing_ok=True)
        _write_new_file(path, data)

    def remove_chunks(self, identifier: str) -> None:
        """Delete every stored part of one session."""
        _remove_entry(self.session_directory(identifier) / CHUNKS_DIRECTORY)

    def verified_parts(
        self, identifier: str, session: DatasetUploadSession
    ) -> list[int]:
        """Keep the received parts whose files have the right size; drop the rest."""
        present: list[int] = []
        for index in session.received_chunks:
            expected = expected_chunk_bytes(session, index)
            if self.chunk_size(identifier, index) == expected:
                present.append(index)
            else:
                self.chunk_path(identifier, index).unlink(missing_ok=True)
        return present

    def assemble(self, identifier: str, chunk_count: int) -> tuple[str, bytes]:
        """Join the parts in order into a new archive, removing each part used.

        Return the digest of the joined bytes and the leading bytes for the
        archive check. A leftover archive from an interrupted attempt is replaced.
        """
        digest = hashlib.sha256()
        prefix = b""
        destination = self.archive_path(identifier)
        destination.unlink(missing_ok=True)
        descriptor = os.open(destination, _OPEN_NEW, OWNER_ONLY_FILE)
        with os.fdopen(descriptor, "wb") as output:
            for index in range(chunk_count):
                part = self.chunk_path(identifier, index)
                for block in _iter_file_blocks(part):
                    digest.update(block)
                    output.write(block)
                    prefix += block[: ARCHIVE_PREFIX_BYTES - len(prefix)]
                part.unlink()
            os.fsync(output.fileno())
        _fsync_directory(destination.parent)
        return digest.hexdigest(), prefix

    def used_bytes(self) -> int:
        """Count the bytes that stored parts and assembled archives occupy."""
        total = 0
        for directory, _names, file_names in os.walk(self.root):
            for name in file_names:
                if name.endswith((".part", ARCHIVE_NAME)):
                    total += _regular_file_size(Path(directory) / name) or 0
        return total
