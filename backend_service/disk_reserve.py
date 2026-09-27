"""Shared free-space reserve rule for every bounded write on the data volume."""

import shutil
from pathlib import Path

from backend_service.dataset_files import dataset_failure

MINIMUM_RESERVE_BYTES = 10 * 1024**3


def minimum_free_bytes(directory: Path) -> int:
    """Return the reserve to keep: at least 10 GiB or one tenth of the disk."""
    try:
        usage = shutil.disk_usage(directory)
    except OSError:
        raise dataset_failure("dataset_storage") from None
    return max(MINIMUM_RESERVE_BYTES, usage.total // 10)


def free_disk_bytes(directory: Path) -> int:
    """Report free bytes on the filesystem that holds the directory."""
    try:
        return shutil.disk_usage(directory).free
    except OSError:
        raise dataset_failure("dataset_storage") from None


def ensure_disk_reserve(directory: Path, next_bytes: int) -> None:
    """Refuse a write that would eat into the reserve."""
    if next_bytes < 0:
        raise dataset_failure("dataset_space")
    try:
        usage = shutil.disk_usage(directory)
    except OSError:
        raise dataset_failure("dataset_storage") from None
    if usage.free - next_bytes < max(MINIMUM_RESERVE_BYTES, usage.total // 10):
        raise dataset_failure("dataset_space")
