"""Decide which archive members are safe images or text before extracting them."""

import os
import struct
from dataclasses import dataclass
from typing import IO, Literal

from backend_service.failures import ApplicationFailure
from schemas.dataset_common import relative_path

MemberKind = Literal["image", "text", "rejected"]
SNIFF_BYTES = 512
PNG_MAGIC = b"\x89PNG"
JPEG_MAGIC = b"\xff\xd8"
ZIP_MAGIC = b"PK\x03\x04"
GZIP_MAGIC = b"\x1f\x8b"
TAR_MAGIC = b"ustar"
TAR_MAGIC_OFFSET = 257
END_RECORD_SIGNATURE = b"PK\x05\x06"
END_RECORD_BYTES = 22
MAXIMUM_COMMENT_BYTES = 65_535
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_LOCATOR_BYTES = 20
ZIP64_RECORD_SIGNATURE = b"PK\x06\x06"
ZIP64_RECORD_BYTES = 56
MAXIMUM_DIRECTORY_BYTES = 64 * 1024**2
IMAGE_EXTENSIONS = {"png": PNG_MAGIC, "jpg": JPEG_MAGIC, "jpeg": JPEG_MAGIC}
TEXT_EXTENSIONS = frozenset({"txt"})
METADATA_FOLDER = "__MACOSX"
REJECTION_REASONS = frozenset(
    {
        "unsafe_path",
        "unsupported_file_type",
        "content_mismatch",
        "nested_archive",
        "member_size_limit",
        "duplicate_member",
        "encrypted_member",
        "special_member",
        "size_mismatch",
    }
)
_MESSAGES = {
    "archive_unreadable": (
        "The archive could not be read. It may be damaged, truncated or not a "
        "zip or tar file."
    ),
    "archive_limits": (
        "The archive holds more files or bytes than allowed. "
        "Choose a smaller archive or raise the limits."
    ),
    "archive_member_size": (
        "A file inside the archive is not the size the archive declared. "
        "The archive was rejected because its contents cannot be trusted."
    ),
}


def archive_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for an archive failure code."""
    status_code = 400 if code == "archive_limits" else 422
    return ApplicationFailure(code, _MESSAGES[code], status_code)


@dataclass(frozen=True)
class MemberPolicy:
    """Limits that bound what an archive may put on disk."""

    allowed_kinds: frozenset[str] = frozenset({"image", "text"})
    maximum_members: int = 200_000
    maximum_total_bytes: int = 100 * 1024**3
    maximum_image_bytes: int = 50 * 1024**2
    maximum_text_bytes: int = 256 * 1024**2

    def maximum_bytes_for(self, kind: str) -> int:
        """Return the per-member byte limit for one accepted kind."""
        return self.maximum_image_bytes if kind == "image" else self.maximum_text_bytes


@dataclass(frozen=True)
class MemberDecision:
    """Whether a member is kept as an image or text, or why it is rejected."""

    kind: MemberKind
    reason: str | None


@dataclass(frozen=True)
class MemberRecord:
    """One member written to the stage with its verified size and digest."""

    path: str
    kind: str
    size: int
    sha256: str
    split_label: str | None


@dataclass(frozen=True)
class MemberRejection:
    """One member that was skipped and the reason it was skipped."""

    path: str
    reason: str
    declared_bytes: int


@dataclass(frozen=True)
class ExtractionReport:
    """Everything an extraction kept, everything it skipped, and the bytes kept."""

    members: tuple[MemberRecord, ...]
    rejected: tuple[MemberRejection, ...]
    total_bytes: int


@dataclass(frozen=True)
class ZipDirectory:
    """What the end of a zip file says about its central directory."""

    entries: int
    size_bytes: int


def looks_like_archive(first_bytes: bytes) -> bool:
    """Tell whether bytes start like a zip, gzip or tar stream."""
    return (
        first_bytes.startswith(ZIP_MAGIC)
        or first_bytes.startswith(GZIP_MAGIC)
        or first_bytes[TAR_MAGIC_OFFSET : TAR_MAGIC_OFFSET + len(TAR_MAGIC)]
        == TAR_MAGIC
    )


def read_zip_directory(stream: IO[bytes]) -> ZipDirectory:
    """Read the entry count and central directory size from the end records.

    Only the end of the file is read, so nothing of the central directory is
    parsed before its size is known. Zip64 archives keep the real numbers in
    the zip64 end record that the locator points at; an archive that shows
    zip64 values without a locator is read like Python reads it, from the
    plain record.
    """
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    start = max(0, size - END_RECORD_BYTES - MAXIMUM_COMMENT_BYTES)
    stream.seek(start)
    tail = stream.read(size - start)
    position = tail.rfind(END_RECORD_SIGNATURE)
    if position < 0 or len(tail) - position < END_RECORD_BYTES:
        raise archive_failure("archive_unreadable")
    record = struct.unpack("<4s4H2LH", tail[position : position + END_RECORD_BYTES])
    entries, directory_size = int(record[4]), int(record[5])
    plain = entries != 0xFFFF and directory_size != 0xFFFFFFFF
    locator_at = position - ZIP64_LOCATOR_BYTES
    if (
        plain
        or locator_at < 0
        or not tail[locator_at:].startswith(ZIP64_LOCATOR_SIGNATURE)
    ):
        return ZipDirectory(entries, directory_size)
    locator = struct.unpack("<4sLQL", tail[locator_at:position])
    stream.seek(int(locator[2]))
    zip64 = stream.read(ZIP64_RECORD_BYTES)
    if len(zip64) != ZIP64_RECORD_BYTES or not zip64.startswith(ZIP64_RECORD_SIGNATURE):
        raise archive_failure("archive_unreadable")
    fields = struct.unpack("<4sQ2H2L4Q", zip64)
    return ZipDirectory(int(fields[7]), int(fields[8]))


def check_zip_directory(stream: IO[bytes], policy: MemberPolicy) -> ZipDirectory:
    """Refuse a zip whose central directory is too big to hold in memory."""
    directory = read_zip_directory(stream)
    if (
        directory.entries > policy.maximum_members
        or directory.size_bytes > MAXIMUM_DIRECTORY_BYTES
    ):
        raise archive_failure("archive_limits")
    return directory


def _is_metadata(name: str) -> bool:
    """Tell whether a member is an operating-system sidecar rather than content."""
    parts = name.split("/")
    return parts[0] == METADATA_FOLDER or any(part.startswith(".") for part in parts)


def _rejected(reason: str) -> MemberDecision:
    """Build a rejection decision for one reason."""
    return MemberDecision("rejected", reason)


def classify_member(
    name: str, first_bytes: bytes, policy: MemberPolicy
) -> MemberDecision:
    """Accept only safely named images and text whose bytes match their name."""
    try:
        relative_path(name)
    except ValueError:
        return _rejected("unsafe_path")
    if _is_metadata(name):
        return _rejected("unsupported_file_type")
    if looks_like_archive(first_bytes):
        return _rejected("nested_archive")
    basename = name.rsplit("/", 1)[-1]
    extension = basename.rsplit(".", 1)[-1].lower() if "." in basename else ""
    if extension in IMAGE_EXTENSIONS:
        kind: MemberKind = "image"
    elif extension in TEXT_EXTENSIONS:
        kind = "text"
    else:
        return _rejected("unsupported_file_type")
    if kind not in policy.allowed_kinds:
        return _rejected("unsupported_file_type")
    if kind == "image" and not first_bytes.startswith(IMAGE_EXTENSIONS[extension]):
        return _rejected("content_mismatch")
    if kind == "text" and b"\x00" in first_bytes:
        return _rejected("content_mismatch")
    return MemberDecision(kind, None)
