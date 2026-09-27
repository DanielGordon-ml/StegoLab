"""Serialized, bounded dataset staging on persistent local storage."""

import fcntl
import hashlib
import os
import stat
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

from backend_service.dataset_files import (
    MAXIMUM_SOURCE_BYTES,
    dataset_failure,
    directory_descriptor,
    relative_parts,
    validate_source_output,
)
from backend_service.dataset_storage_ops import (
    create_owner_marker,
    flush_directory_tree,
    publish_directory,
    recover_staging,
    remove_owned_stage,
)
from backend_service.disk_reserve import ensure_disk_reserve


class DatasetWriter:
    """Own a root lock and one same-filesystem stage until publication/cleanup."""

    def __init__(
        self,
        output_root: Path,
        source_directory: Path,
        maximum_prepared_bytes: int = MAXIMUM_SOURCE_BYTES,
    ) -> None:
        """Set storage policy without creating files or reading source pixels."""
        self.output_root = Path(os.path.abspath(output_root))
        self.source_directory = source_directory
        self.maximum_prepared_bytes = maximum_prepared_bytes
        self.authorized_bytes = 0
        self.stage = self.output_root / ".staging" / f"run-{uuid.uuid4().hex}"
        self.marker = self.stage.with_name(self.stage.name + ".owner.json")
        self._lock: int | None = None
        self._published = False

    def __enter__(self) -> "DatasetWriter":
        """Validate roots, acquire the exclusive lock, and recover owned stages."""
        validate_source_output(self.source_directory, self.output_root)
        try:
            with directory_descriptor(self.output_root, create=True) as root:
                self._lock = os.open(
                    ".dataset.lock",
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                    dir_fd=root,
                )
            if not stat.S_ISREG(os.fstat(self._lock).st_mode):
                raise dataset_failure("dataset_storage")
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise dataset_failure("dataset_busy") from None
            with directory_descriptor(self.stage.parent, create=True):
                pass
            recover_staging(self.stage.parent)
            self._check_free_space(1024)
            create_owner_marker(self.stage, self.marker)
            self.stage.mkdir(mode=0o700)
            with directory_descriptor(self.stage.parent) as staging:
                os.fsync(staging)
            return self
        except BaseException as failure:
            try:
                if self.marker.exists():
                    remove_owned_stage(self.stage, self.marker)
                elif self.stage.exists():
                    self.stage.rmdir()
            finally:
                self._release_lock()
            if isinstance(failure, OSError):
                raise dataset_failure("dataset_storage") from None
            raise

    def _release_lock(self) -> None:
        """Close this writer's descriptor and release its root lock."""
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Keep published revisions and remove only this run's marked stage."""
        try:
            if self.marker.exists():
                remove_owned_stage(self.stage, self.marker)
        finally:
            self._release_lock()

    def check_estimated_space(self, estimated_bytes: int) -> None:
        """Check estimated next writes without charging the run's byte allowance."""
        if (
            estimated_bytes < 0
            or self.authorized_bytes + estimated_bytes > self.maximum_prepared_bytes
        ):
            raise dataset_failure("dataset_space")
        self._check_free_space(estimated_bytes)

    def _check_free_space(self, next_bytes: int) -> None:
        """Preserve the shared disk reserve before any bounded write."""
        ensure_disk_reserve(self.output_root, next_bytes)

    def reserve_write(self, size: int) -> None:
        """Charge each bounded write before it reaches the destination filesystem."""
        if self._lock is None or self._published:
            raise dataset_failure("dataset_storage")
        self.check_estimated_space(size)
        self.authorized_bytes += size

    def write_bytes(self, relative_path: str, data: bytes) -> Path:
        """Write a new small record within this stage without replacing entries."""
        parts = relative_parts(relative_path)
        path = self.stage.joinpath(*parts)
        try:
            with directory_descriptor(path.parent, create=True) as parent:
                descriptor = os.open(
                    parts[-1],
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent,
                )
                with os.fdopen(descriptor, "wb") as output:
                    for offset in range(0, len(data), 1024**2):
                        chunk = data[offset : offset + 1024**2]
                        self.reserve_write(len(chunk))
                        output.write(chunk)
                    os.fsync(output.fileno())
                os.fsync(parent)
            return path
        except OSError:
            raise dataset_failure("dataset_storage") from None

    def write_stream(
        self, relative_path: str, chunks: Iterator[bytes], *, maximum_bytes: int
    ) -> tuple[str, int]:
        """Stream a new file into this stage, charging bytes as they arrive."""
        parts = relative_parts(relative_path)
        path = self.stage.joinpath(*parts)
        digest = hashlib.sha256()
        total = 0
        try:
            with directory_descriptor(path.parent, create=True) as parent:
                descriptor = os.open(
                    parts[-1],
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent,
                )
                with os.fdopen(descriptor, "wb") as output:
                    for chunk in chunks:
                        total += len(chunk)
                        if total > maximum_bytes:
                            raise dataset_failure("dataset_limits")
                        self.reserve_write(len(chunk))
                        digest.update(chunk)
                        output.write(chunk)
                    os.fsync(output.fileno())
                os.fsync(parent)
        except OSError:
            raise dataset_failure("dataset_storage") from None
        return digest.hexdigest(), total

    def record_external_write(self, relative_path: str, size: int) -> None:
        """Check a callback-budgeted file without charging its bytes twice."""
        path = self.stage.joinpath(*relative_parts(relative_path))
        try:
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size != size
                or size > self.authorized_bytes
            ):
                raise dataset_failure("dataset_storage")
        except OSError:
            raise dataset_failure("dataset_storage") from None

    def reuse_path(self, relative_directory: str) -> Path | None:
        """Return an existing safe revision path for full integrity validation."""
        path = self.output_root.joinpath(*relative_parts(relative_directory))
        if path.exists() or path.is_symlink():
            with directory_descriptor(path):
                pass
            return path
        return None

    def publish(self, relative_directory: str) -> Path:
        """Publish this complete stage once, without replacing prior revisions."""
        if self._lock is None or self._published:
            raise dataset_failure("dataset_storage")
        path = self.output_root.joinpath(*relative_parts(relative_directory))
        try:
            with directory_descriptor(path.parent, create=True) as parent:
                self.check_estimated_space(0)
                flush_directory_tree(self.stage)
                publish_directory(self.stage, path)
                self._published = True
                os.fsync(parent)
            self.marker.unlink()
            with directory_descriptor(self.stage.parent) as staging:
                os.fsync(staging)
            return path
        except OSError:
            raise dataset_failure("dataset_storage") from None
