"""Cache layout, lock, records and small durable file helpers."""

import fcntl
import logging
import os
import re
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from backend_service.dataset_files import directory_descriptor
from backend_service.dataset_serialization import (
    canonical_json,
    file_checksum,
    read_bounded,
)
from backend_service.dataset_sources.transfer import PartialFile
from backend_service.failures import ApplicationFailure
from schemas.dataset_cache import CacheEntryManifest, CacheKind, CachePartialRecord

logger = logging.getLogger(__name__)

MANIFEST_NAME = "cache_manifest.json"
PARTIAL_NAME = "partial.json"
LOCK_NAME = ".cache.lock"
PARTIAL_SUFFIX = ".partial"
PART_SUFFIX = ".part"
MAXIMUM_RECORD_BYTES = 64 * 1024
MAXIMUM_SUMMARY_ENTRIES = 1000
RESERVED_PREFIX = "asset_"
RESERVED_NAMES = frozenset({MANIFEST_NAME, PARTIAL_NAME, LOCK_NAME})
CACHE_KINDS: tuple[CacheKind, ...] = ("hugging_face", "https_archive", "upload")
IDENTITY = re.compile(r"[a-f0-9]{64}")
_MESSAGES = {
    "cache_storage": (
        "The download cache could not be read or written. "
        "Check access and free space, then retry."
    ),
    "cache_key": "The cache key is not valid. This is an internal error.",
    "cache_checksum_mismatch": (
        "The downloaded file does not match its expected checksum. "
        "The partial download was removed; retry the fetch."
    ),
    "cache_size_mismatch": (
        "The downloaded file does not have its expected size. "
        "The partial download was removed; retry the fetch."
    ),
    "cache_too_large": "The file to import is larger than the allowed size.",
    "cache_entry_corrupt": (
        "A cached file is damaged. Clear the download cache and retry."
    ),
}


def cache_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for a cache failure code."""
    status_code = 400 if code == "cache_too_large" else 502
    return ApplicationFailure(code, _MESSAGES[code], status_code)


def cache_kind(value: str) -> CacheKind:
    """Narrow a source kind to one the cache stores."""
    for kind in CACHE_KINDS:
        if kind == value:
            return kind
    raise cache_failure("cache_key")


def is_reserved_name(name: str) -> bool:
    """Tell whether a file name would collide with the cache's own records.

    The record names and their temporary forms are reserved; the prefix that
    ``cache_basename`` adds makes any such name safe again.
    """
    return name in RESERVED_NAMES or name.startswith((MANIFEST_NAME, PARTIAL_NAME))


def cache_basename(path: str) -> str:
    """Turn the last path segment into a plain file name the manifest accepts.

    Names the cache uses for its own records get a fixed prefix so an asset
    can never overwrite a record or be mistaken for one.
    """
    simplified = re.sub(r"[^A-Za-z0-9._-]", "_", path.rsplit("/", 1)[-1])
    simplified = (simplified.lstrip("._-") or "asset")[:255]
    if is_reserved_name(simplified):
        simplified = (RESERVED_PREFIX + simplified)[:255]
    return simplified


def current_timestamp() -> str:
    """Timestamp partial records; never used for identities."""
    return datetime.now(UTC).isoformat()


def cache_directory(root: Path, kind: str, identity: str, *, partial: bool) -> Path:
    """Locate the complete or partial directory for one asset identity."""
    if IDENTITY.fullmatch(identity) is None:
        raise cache_failure("cache_key")
    return root / cache_kind(kind) / (identity + (PARTIAL_SUFFIX if partial else ""))


@contextmanager
def cache_lock(root: Path) -> Iterator[None]:
    """Hold the exclusive cache lock while metadata changes; never nest it."""
    with directory_descriptor(root, create=True) as directory:
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
        descriptor = os.open(LOCK_NAME, flags, 0o600, dir_fd=directory)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise cache_failure("cache_storage")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def read_manifest(directory: Path) -> CacheEntryManifest | None:
    """Parse a complete entry's manifest strictly, or return None when unusable."""
    try:
        data = read_bounded(directory / MANIFEST_NAME, MAXIMUM_RECORD_BYTES)
        return CacheEntryManifest.model_validate_json(data)
    except (OSError, ValueError, ApplicationFailure):
        return None


def read_partial_record(directory: Path) -> CachePartialRecord | None:
    """Parse a partial directory's record strictly, or return None when unusable."""
    try:
        data = read_bounded(directory / PARTIAL_NAME, MAXIMUM_RECORD_BYTES)
        return CachePartialRecord.model_validate_json(data)
    except (OSError, ValueError, ApplicationFailure):
        return None


def write_record(directory: Path, name: str, data: bytes, *, replace: bool) -> None:
    """Write one small record durably: new-only, or through an atomic replace."""
    target = f"{name}.tmp" if replace else name
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    flags |= os.O_TRUNC if replace else os.O_EXCL
    try:
        with directory_descriptor(directory) as parent:
            descriptor = os.open(target, flags, 0o600, dir_fd=parent)
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                os.fsync(output.fileno())
            if replace:
                os.replace(target, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
    except OSError:
        raise cache_failure("cache_storage") from None


def write_manifest_file(directory: Path, manifest: CacheEntryManifest) -> None:
    """Write the entry manifest once, after the data file is complete."""
    data = canonical_json(manifest.model_dump(mode="json"))
    write_record(directory, MANIFEST_NAME, data, replace=False)


def write_partial_record(directory: Path, record: CachePartialRecord) -> None:
    """Replace partial.json atomically with the given record."""
    data = canonical_json(record.model_dump(mode="json"))
    write_record(directory, PARTIAL_NAME, data, replace=True)


def remove_tree(path: Path) -> None:
    """Delete one cache directory (or a stray symlink) without following links."""
    try:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except OSError:
        raise cache_failure("cache_storage") from None


def is_regular(path: Path) -> bool:
    """Tell whether the path is a regular file that is not a symlink."""
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


@dataclass(frozen=True)
class CacheEntry:
    """One verified cached file and the manifest that describes it."""

    kind: str
    identity: str
    directory: Path
    file: Path
    manifest: CacheEntryManifest


@dataclass
class CachePartial:
    """One unfinished download owned by a job, with its resumable file."""

    kind: str
    identity: str
    directory: Path
    file: PartialFile
    record: CachePartialRecord
    path: str = ""


def load_entry(
    directory: Path, kind: str, identity: str, *, verify_bytes: bool
) -> CacheEntry | None:
    """Validate a complete entry's manifest, size and optionally its checksum."""
    if directory.is_symlink() or not directory.is_dir():
        return None
    manifest = read_manifest(directory)
    if manifest is not None and (manifest.kind, manifest.identity) == (kind, identity):
        file = directory / manifest.basename
        try:
            valid = is_regular(file) and file.lstat().st_size == manifest.size_bytes
            if valid and verify_bytes:
                valid = file_checksum(file, manifest.size_bytes)[0] == manifest.sha256
        except (OSError, ValueError):
            valid = False
        if valid:
            return CacheEntry(kind, identity, directory, file, manifest)
    logger.warning("cache_entry_invalid")
    return None


def scan_cache(root: Path) -> Iterator[tuple[CacheKind, str, Path, bool]]:
    """Yield (kind, name, path, is_partial) for every well-named cache directory.

    The walk is bounded by the directory listings themselves and never fails
    on a large cache; listings that must stay short truncate on their side.
    """
    for kind in CACHE_KINDS:
        base = root / kind
        if base.is_symlink() or not base.is_dir():
            continue
        with os.scandir(base) as entries:
            listing = sorted(entries, key=lambda entry: entry.name)
        for entry in listing:
            if entry.is_symlink() or not entry.is_dir():
                continue
            is_partial = entry.name.endswith(PARTIAL_SUFFIX)
            stem = entry.name[: -len(PARTIAL_SUFFIX)] if is_partial else entry.name
            if IDENTITY.fullmatch(stem) is None:
                continue
            yield kind, entry.name, Path(entry.path), is_partial
