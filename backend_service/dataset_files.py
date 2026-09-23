"""Bounded local file access with anchored, no-follow directory traversal."""

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from backend_service.failures import ApplicationFailure

MAXIMUM_SOURCE_BYTES = 100 * 1024**3
MAXIMUM_FILE_BYTES = 50 * 1024**2


def dataset_failure(code: str = "dataset_access") -> ApplicationFailure:
    """Return fixed error messages without paths or source metadata."""
    messages = {
        "dataset_empty": (
            "The selected source has no image candidates. Add JPEG or PNG files."
        ),
        "dataset_access": (
            "Dataset files could not be accessed safely. Check access and paths."
        ),
        "dataset_limits": (
            "The dataset exceeds its file or byte limits. "
            "Select a smaller named subset."
        ),
        "dataset_changed": (
            "A source file changed during preparation. Retry with a stable source."
        ),
        "dataset_metadata": (
            "UHD-IQA metadata is missing, ambiguous, or invalid. "
            "Check names and splits."
        ),
        "dataset_storage": (
            "Dataset output could not be saved. Check access and free space."
        ),
        "dataset_space": (
            "Dataset preparation would exceed its output limit or free-space reserve."
        ),
        "dataset_busy": (
            "Another dataset preparation is using this output root. "
            "Retry when it finishes."
        ),
        "dataset_exists": (
            "This dataset revision already exists. Validate and reuse it."
        ),
    }
    return ApplicationFailure(code, messages[code], 400)


def relative_parts(relative_path: str) -> tuple[str, ...]:
    """Accept only ordinary relative POSIX paths without ambiguous components."""
    parts = PurePosixPath(relative_path).parts
    if (
        not parts
        or relative_path != "/".join(parts)
        or any(part in ("", ".", "..") for part in parts)
        or "\\" in relative_path
        or ":" in relative_path
        or len(relative_path) > 4096
        or any(
            ord(character) < 32 or ord(character) == 127 for character in relative_path
        )
        or PurePosixPath(relative_path).is_absolute()
    ):
        raise dataset_failure()
    return parts


@contextmanager
def directory_descriptor(path: Path, *, create: bool = False) -> Iterator[int]:
    """Open each directory from filesystem root, refusing symlink parents."""
    absolute = Path(os.path.abspath(path))
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    except OSError:
        raise dataset_failure() from None
    finally:
        os.close(descriptor)


def validate_source_output(source_directory: Path, output_root: Path) -> None:
    """Reject source/output overlap before creating output directories."""
    source = Path(os.path.abspath(source_directory))
    output = Path(os.path.abspath(output_root))
    if source == output or source in output.parents or output in source.parents:
        raise dataset_failure()
    with directory_descriptor(source):
        pass
    existing = output
    while not existing.exists() and not existing.is_symlink():
        existing = existing.parent
    with directory_descriptor(existing):
        pass


def stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """Identify a file and its content-bearing timestamps without reading it."""
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclass(frozen=True)
class SourceFile:
    """Freeze one discovered source entry for later safe bounded reads."""

    relative_path: str
    size_bytes: int
    identity: tuple[int, int, int, int, int]
    kind: str
    rejection_reason: str | None = None


def read_source_bytes(
    root: Path, source: SourceFile, limit: int = MAXIMUM_FILE_BYTES
) -> bytes:
    """Read the same scanned object and reject replacement or modification."""
    return b"".join(_source_chunks(root, source, limit))


def hash_source_file(
    root: Path, source: SourceFile, limit: int = MAXIMUM_SOURCE_BYTES
) -> str:
    """Hash rejected or oversized source files with one bounded read buffer."""
    digest = hashlib.sha256()
    for chunk in _source_chunks(root, source, limit):
        digest.update(chunk)
    return digest.hexdigest()


def _source_chunks(root: Path, source: SourceFile, limit: int) -> Iterator[bytes]:
    """Yield bounded bytes from the same safely opened source object."""
    parts = relative_parts(source.relative_path)
    try:
        with directory_descriptor(root.joinpath(*parts[:-1])) as parent:
            descriptor = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise dataset_failure()
                if stat_identity(before) != source.identity:
                    raise dataset_failure("dataset_changed")
                if before.st_size > limit:
                    raise dataset_failure("dataset_limits")
                total = 0
                while True:
                    chunk = os.read(descriptor, min(1024**2, limit - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise dataset_failure("dataset_limits")
                    yield chunk
                after = os.fstat(descriptor)
                current = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                if (
                    stat_identity(after) != source.identity
                    or stat_identity(current) != source.identity
                    or total != source.size_bytes
                ):
                    raise dataset_failure("dataset_changed")
            finally:
                os.close(descriptor)
    except PermissionError:
        raise dataset_failure() from None
    except OSError:
        raise dataset_failure("dataset_changed") from None
