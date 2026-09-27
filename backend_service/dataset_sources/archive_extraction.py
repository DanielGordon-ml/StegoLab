"""Stream zip and tar members into a dataset stage under strict limits."""

import logging
import os
import stat
import tarfile
import zipfile
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

from backend_service.dataset_sources.archive_members import (
    GZIP_MAGIC,
    SNIFF_BYTES,
    ZIP_MAGIC,
    ExtractionReport,
    MemberPolicy,
    MemberRecord,
    MemberRejection,
    archive_failure,
    check_zip_directory,
    classify_member,
    looks_like_archive,
)
from backend_service.dataset_sources.transfer import DownloadStopped
from backend_service.dataset_storage import DatasetWriter
from backend_service.failures import ApplicationFailure

logger = logging.getLogger(__name__)

STREAM_CHUNK_BYTES = 1024**2
SYMLINK_MODE = 0o120000
ENCRYPTED_FLAG = 0x1
ARCHIVE_ERRORS = (zipfile.BadZipFile, tarfile.TarError, zlib.error, EOFError, OSError)
SplitLookup = Callable[[str], str | None]
ProgressCallback = Callable[[int, int | None], None]


@dataclass(frozen=True)
class ArchiveMember:
    """One archive entry, described the same way for zip and tar.

    A ``streamed`` member (tar) can only be skipped by reading through its
    bytes, so its declared size is bounded before anything is read.
    """

    name: str
    declared_bytes: int
    special: bool
    encrypted: bool
    open: Callable[[], IO[bytes]]
    streamed: bool = False


@dataclass
class _Extraction:
    """Mutable bookkeeping shared by every member of one extraction run."""

    writer: DatasetWriter
    policy: MemberPolicy
    split_for_member: SplitLookup
    progress: ProgressCallback | None
    stop: Callable[[], bool]
    total_declared: int | None
    seen: set[str] = field(default_factory=set)
    members: list[MemberRecord] = field(default_factory=list)
    rejected: list[MemberRejection] = field(default_factory=list)
    kept_bytes: int = 0
    streamed_bytes: int = 0
    count: int = 0

    def check_stop(self) -> None:
        """Raise ``DownloadStopped`` when the caller asked to stop."""
        if self.stop():
            logger.info("extraction_stopped")
            raise DownloadStopped()

    def reject(self, member: ArchiveMember, reason: str) -> None:
        """Record one skipped member without failing the run."""
        self.rejected.append(
            MemberRejection(member.name, reason, max(member.declared_bytes, 0))
        )

    def _check_streamed(self, member: ArchiveMember) -> None:
        """Bound the bytes a streamed archive makes the run read through.

        A tar member is skipped by reading its body, so a member larger than
        any kind may keep, or a declared total beyond the byte limit, fails
        the run before that reading starts.
        """
        if not member.streamed:
            return
        largest = max(self.policy.maximum_image_bytes, self.policy.maximum_text_bytes)
        self.streamed_bytes += max(member.declared_bytes, 0)
        if (
            member.declared_bytes > largest
            or self.streamed_bytes > self.policy.maximum_total_bytes
        ):
            logger.warning("archive_streamed_member_too_large")
            raise archive_failure("archive_limits")

    def handle(self, member: ArchiveMember) -> None:
        """Check one member against every rule, then stream it when accepted."""
        self.check_stop()
        self.count += 1
        if self.count > self.policy.maximum_members:
            raise archive_failure("archive_limits")
        self._check_streamed(member)
        if member.special:
            return self.reject(member, "special_member")
        if member.encrypted:
            return self.reject(member, "encrypted_member")
        if member.name in self.seen:
            return self.reject(member, "duplicate_member")
        self.seen.add(member.name)
        stream = member.open()
        try:
            head = stream.read(SNIFF_BYTES)
            decision = classify_member(member.name, head, self.policy)
            if decision.reason is not None:
                return self.reject(member, decision.reason)
            if member.declared_bytes > self.policy.maximum_bytes_for(decision.kind):
                return self.reject(member, "member_size_limit")
            if (
                self.kept_bytes + member.declared_bytes
                > self.policy.maximum_total_bytes
            ):
                raise archive_failure("archive_limits")
            digest, size = self._stream(member, head, stream)
        finally:
            stream.close()
        self.kept_bytes += size
        self.members.append(
            MemberRecord(
                member.name,
                decision.kind,
                size,
                digest,
                self.split_for_member(member.name),
            )
        )
        if self.progress is not None:
            self.progress(self.kept_bytes, self.total_declared)

    def _stream(
        self, member: ArchiveMember, head: bytes, stream: IO[bytes]
    ) -> tuple[str, int]:
        """Write the member through the stage and insist on the declared size."""

        def chunks() -> Iterator[bytes]:
            """Yield the head and the rest, checking for a stop between chunks."""
            yield head
            chunk = stream.read(STREAM_CHUNK_BYTES)
            while chunk:
                following = stream.read(STREAM_CHUNK_BYTES)
                yield chunk
                if following:
                    self.check_stop()
                chunk = following

        try:
            digest, size = self.writer.write_stream(
                member.name, chunks(), maximum_bytes=member.declared_bytes
            )
        except ApplicationFailure as failure:
            if failure.code != "dataset_limits":
                raise
            size = member.declared_bytes + 1
        except ARCHIVE_ERRORS:
            raise archive_failure("archive_unreadable") from None
        if size != member.declared_bytes:
            self.reject(member, "size_mismatch")
            logger.warning("archive_member_size_lie")
            raise archive_failure("archive_member_size")
        return digest, size


def _zip_members(archive: zipfile.ZipFile) -> Iterator[ArchiveMember]:
    """Describe every file entry of a zip archive, skipping directory entries."""

    def opener(info: zipfile.ZipInfo) -> Callable[[], IO[bytes]]:
        """Bind one entry so it can be opened later without password handling."""
        return lambda: archive.open(info)

    for info in archive.infolist():
        if info.is_dir():
            continue
        mode = (info.external_attr >> 16) & 0o170000
        yield ArchiveMember(
            name=info.filename,
            declared_bytes=info.file_size,
            special=mode == SYMLINK_MODE,
            encrypted=bool(info.flag_bits & ENCRYPTED_FLAG),
            open=opener(info),
        )


def _tar_members(archive: tarfile.TarFile) -> Iterator[ArchiveMember]:
    """Describe every entry of a streamed tar archive, skipping directories."""

    def opener(member: tarfile.TarInfo) -> Callable[[], IO[bytes]]:
        """Open one regular tar member, reporting a damaged stream plainly."""

        def open_member() -> IO[bytes]:
            """Return the member's readable stream."""
            stream = archive.extractfile(member)
            if stream is None:
                raise archive_failure("archive_unreadable")
            return stream

        return open_member

    for member in archive:
        if member.isdir():
            continue
        yield ArchiveMember(
            name=member.name,
            declared_bytes=member.size,
            special=not member.isreg(),
            encrypted=False,
            open=opener(member),
            streamed=True,
        )


def _open_archive(archive: Path) -> IO[bytes]:
    """Open a regular archive file for reading without following links."""
    try:
        descriptor = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        raise archive_failure("archive_unreadable") from None
    stream = os.fdopen(descriptor, "rb")
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        stream.close()
        raise archive_failure("archive_unreadable")
    return stream


def _extract_zip(stream: IO[bytes], state: _Extraction) -> None:
    """Bound the central directory from the end records, then walk the entries."""
    check_zip_directory(stream, state.policy)
    with zipfile.ZipFile(stream) as zipped:
        infos = [info for info in zipped.infolist() if not info.is_dir()]
        if len(infos) > state.policy.maximum_members:
            raise archive_failure("archive_limits")
        state.total_declared = sum(info.file_size for info in infos)
        for member in _zip_members(zipped):
            state.handle(member)


def _extract_tar(stream: IO[bytes], head: bytes, state: _Extraction) -> None:
    """Walk a plain or gzip tar stream one member at a time."""
    mode: Literal["r|gz", "r|"] = "r|gz" if head.startswith(GZIP_MAGIC) else "r|"
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode=mode) as tarred:
        for member in _tar_members(tarred):
            state.handle(member)


def extract_archive(
    archive: Path,
    writer: DatasetWriter,
    *,
    policy: MemberPolicy,
    split_for_member: SplitLookup,
    progress: ProgressCallback | None = None,
    stop: Callable[[], bool] = lambda: False,
) -> ExtractionReport:
    """Stream accepted members into the writer's stage and report the rest.

    Rejected members are recorded, not fatal. A member whose bytes differ from
    its declared size, an unreadable archive, or a breached limit fails the
    run; a tar member too large for any kind fails it too, because skipping
    it would mean reading it. ``DownloadStopped`` is raised when ``stop``
    returns true between members or between the chunks of a large member.
    """
    state = _Extraction(writer, policy, split_for_member, progress, stop, None)
    try:
        with _open_archive(archive) as stream:
            head = stream.read(SNIFF_BYTES)
            if not looks_like_archive(head):
                raise archive_failure("archive_unreadable")
            if head.startswith(ZIP_MAGIC):
                _extract_zip(stream, state)
            else:
                _extract_tar(stream, head, state)
    except ARCHIVE_ERRORS:
        raise archive_failure("archive_unreadable") from None
    logger.info("archive_extracted")
    return ExtractionReport(
        members=tuple(state.members),
        rejected=tuple(state.rejected),
        total_bytes=state.kept_bytes,
    )
