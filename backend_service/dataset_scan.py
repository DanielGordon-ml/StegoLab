"""Deterministic bounded discovery of extracted local dataset files."""

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from backend_service.dataset_files import (
    MAXIMUM_FILE_BYTES,
    MAXIMUM_SOURCE_BYTES,
    SourceFile,
    dataset_failure,
    directory_descriptor,
    relative_parts,
    stat_identity,
)


@dataclass(frozen=True)
class SourceInventory:
    """Describe all scanned files, including rejected and metadata entries."""

    files: tuple[SourceFile, ...]
    scanned_entries: int
    source_bytes: int


def scan_source(
    source_directory: Path,
    *,
    maximum_entries: int = 200_000,
    maximum_source_bytes: int = MAXIMUM_SOURCE_BYTES,
    maximum_file_bytes: int = MAXIMUM_FILE_BYTES,
) -> SourceInventory:
    """Count every entry/regular-file byte and reject unsafe traversal."""
    files: list[SourceFile] = []
    entries = 0
    source_bytes = 0

    def visit(descriptor: int, prefix: str, depth: int) -> None:
        """Count and inspect one directory through an anchored descriptor."""
        nonlocal entries, source_bytes
        if depth > 64:
            raise dataset_failure("dataset_limits")
        with os.scandir(descriptor) as children:
            for child in children:
                entries += 1
                if entries > maximum_entries:
                    raise dataset_failure("dataset_limits")
                relative = f"{prefix}/{child.name}" if prefix else child.name
                relative_parts(relative)
                info = child.stat(follow_symlinks=False)
                hidden = child.name.startswith(".")
                if stat.S_ISDIR(info.st_mode):
                    if hidden:
                        continue
                    nested = os.open(
                        child.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                    try:
                        if stat_identity(os.fstat(nested)) != stat_identity(info):
                            raise dataset_failure("dataset_changed")
                        visit(nested, relative, depth + 1)
                    finally:
                        os.close(nested)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise dataset_failure()
                source_bytes += info.st_size
                if source_bytes > maximum_source_bytes:
                    raise dataset_failure("dataset_limits")
                suffix = Path(child.name).suffix.lower()
                sidecar = (
                    hidden
                    or "__MACOSX" in Path(relative).parts
                    or suffix in (".csv", ".tags")
                )
                kind = "sidecar" if sidecar else "image"
                reason = None
                if not sidecar and suffix not in (".png", ".jpg", ".jpeg"):
                    kind, reason = "unsupported", "unsupported_file_type"
                if info.st_size > maximum_file_bytes:
                    reason = "file_size_limit"
                files.append(
                    SourceFile(
                        relative, info.st_size, stat_identity(info), kind, reason
                    )
                )

    try:
        with directory_descriptor(source_directory) as root:
            visit(root, "", 0)
    except OSError:
        raise dataset_failure() from None
    return SourceInventory(
        tuple(sorted(files, key=lambda entry: entry.relative_path)),
        entries,
        source_bytes,
    )
