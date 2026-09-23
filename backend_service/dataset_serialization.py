"""Canonical UTF-8 dataset records and bounded, contained reads."""

import hashlib
import json
import os
import stat
from pathlib import Path

from backend_service.dataset_files import directory_descriptor
from schemas.base import StrictRecord
from schemas.dataset_common import MAXIMUM_DATASET_ENTRIES, relative_path

MAXIMUM_RECORD_LINE = 64 * 1024
MAXIMUM_METADATA_BYTES = 256 * 1024**2


def canonical_json(value: object) -> bytes:
    """Encode deterministic JSON with one terminal LF and no nonfinite values."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def checksum(data: bytes) -> str:
    """Return the SHA-256 checksum used throughout the dataset format."""
    return hashlib.sha256(data).hexdigest()


def contained_file(directory: Path, relative: str) -> Path:
    """Require every component below the revision to be real and contained."""
    relative_path(relative)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("The dataset revision must be a regular directory.")
    candidate = directory
    for part in relative.split("/"):
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Dataset symlinks are unsupported.")
    if not candidate.is_file():
        raise ValueError("A dataset file is missing or is not regular.")
    if not candidate.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Dataset file escapes its revision.")
    return candidate


def read_bounded(path: Path, maximum: int = MAXIMUM_METADATA_BYTES) -> bytes:
    """Read regular bytes without following a final-component symlink."""
    descriptor = _open_regular_file(path)
    with os.fdopen(descriptor, "rb") as stream:
        information = os.fstat(stream.fileno())
        if not stat.S_ISREG(information.st_mode) or information.st_size > maximum:
            raise ValueError("Dataset file exceeds its size or type limit.")
        data = stream.read(maximum + 1)
        if len(data) > maximum or len(data) != information.st_size:
            raise ValueError("Dataset file changed or exceeds its size limit.")
        return data


def file_checksum(path: Path, maximum: int) -> tuple[str, int]:
    """Hash a bounded regular file in chunks without buffering all its bytes."""
    descriptor = _open_regular_file(path)
    digest = hashlib.sha256()
    count = 0
    with os.fdopen(descriptor, "rb") as stream:
        information = os.fstat(stream.fileno())
        if not stat.S_ISREG(information.st_mode) or information.st_size > maximum:
            raise ValueError("Dataset file exceeds its size or type limit.")
        while block := stream.read(1024 * 1024):
            count += len(block)
            if count > maximum:
                raise ValueError("Dataset file exceeds its size limit.")
            digest.update(block)
        if count != information.st_size:
            raise ValueError("Dataset file changed during verification.")
    return digest.hexdigest(), count


def read_records[Record: StrictRecord](
    directory: Path, name: str, model: type[Record]
) -> list[Record]:
    """Parse bounded canonical JSONL records with strict contracts."""
    path = contained_file(directory, name)
    descriptor = _open_regular_file(path)
    records: list[Record] = []
    total = 0
    with os.fdopen(descriptor, "rb") as stream:
        information = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(information.st_mode)
            or information.st_size > MAXIMUM_METADATA_BYTES
        ):
            raise ValueError("Dataset record file exceeds its type or size limit.")
        while line := stream.readline(MAXIMUM_RECORD_LINE + 1):
            total += len(line)
            if (
                len(line) > MAXIMUM_RECORD_LINE
                or total > MAXIMUM_METADATA_BYTES
                or len(records) >= MAXIMUM_DATASET_ENTRIES
            ):
                raise ValueError("Dataset record count or byte limit exceeded.")
            record = model.model_validate_json(line)
            if canonical_json(record.model_dump(mode="json")) != line:
                raise ValueError("Dataset records are not canonical JSONL.")
            records.append(record)
        if total != information.st_size:
            raise ValueError("Dataset records changed during verification.")
    return records


def _open_regular_file(path: Path) -> int:
    """Anchor every component before opening the final no-follow file."""
    with directory_descriptor(path.parent) as parent:
        return os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
