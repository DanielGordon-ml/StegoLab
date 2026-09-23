"""Portable exclusive publication and ownership-checked staging recovery."""

import ctypes
import errno
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path

from backend_service.dataset_files import dataset_failure, directory_descriptor

RUN_NAME = re.compile(r"run-[0-9a-f]{32}")


def create_owner_marker(stage: Path, marker: Path) -> None:
    """Publish complete ownership bytes before a staging directory can exist."""
    pending = marker.with_name(marker.name + ".pending")
    payload = json.dumps(
        {"version": 1, "owner_uid": os.getuid(), "stage": stage.name}
    ).encode()
    try:
        with pending.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(pending, marker, follow_symlinks=False)
    finally:
        pending.unlink(missing_ok=True)
    with directory_descriptor(stage.parent) as parent:
        os.fsync(parent)


def publish_directory(source: Path, destination: Path) -> None:
    """Rename a complete directory atomically without replacing any entry."""
    library = ctypes.CDLL(None, use_errno=True)
    with (
        directory_descriptor(source.parent) as source_parent,
        directory_descriptor(destination.parent) as destination_parent,
    ):
        if sys.platform == "darwin":
            function, flags = library.renameatx_np, 4
        elif sys.platform.startswith("linux"):
            function, flags = library.renameat2, 1
        else:
            raise dataset_failure("dataset_storage")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = function(
            source_parent,
            os.fsencode(source.name),
            destination_parent,
            os.fsencode(destination.name),
            flags,
        )
    if result != 0:
        code = (
            "dataset_exists"
            if ctypes.get_errno() == errno.EEXIST
            else "dataset_storage"
        )
        raise dataset_failure(code)


def remove_owned_stage(stage: Path, marker: Path) -> None:
    """Remove only a marked staging directory owned by the current user."""
    try:
        descriptor = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_size > 1024
            ):
                raise dataset_failure("dataset_storage")
            metadata = json.loads(os.read(descriptor, 1025))
        finally:
            os.close(descriptor)
        if metadata != {"version": 1, "owner_uid": os.getuid(), "stage": stage.name}:
            raise dataset_failure("dataset_storage")
        if stage.exists() or stage.is_symlink():
            info = stage.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise dataset_failure("dataset_storage")
            shutil.rmtree(stage)
        marker.unlink()
    except (OSError, ValueError):
        raise dataset_failure("dataset_storage") from None


def recover_staging(staging: Path) -> None:
    """Recover owned marked runs only after the caller holds the root lock."""
    with os.scandir(staging) as entries:
        for number, entry in enumerate(entries, start=1):
            if number > 200_000:
                raise dataset_failure("dataset_limits")
            pending_suffix = ".owner.json.pending"
            if entry.name.endswith(pending_suffix):
                name = entry.name[: -len(pending_suffix)]
                if RUN_NAME.fullmatch(name) is None:
                    continue
                info = entry.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_size > 1024
                    or (staging / name).exists()
                ):
                    raise dataset_failure("dataset_storage")
                Path(entry.path).unlink()
                continue
            suffix = ".owner.json"
            if not entry.name.endswith(suffix):
                continue
            name = entry.name[: -len(suffix)]
            if RUN_NAME.fullmatch(name) is None:
                continue
            remove_owned_stage(staging / name, Path(entry.path))


def flush_directory_tree(root: Path) -> None:
    """Persist created directory entries before their containing publication."""
    for path, directories, _files in os.walk(root, followlinks=False):
        for directory in directories:
            if (Path(path) / directory).is_symlink():
                raise dataset_failure("dataset_storage")
        with directory_descriptor(Path(path)) as descriptor:
            os.fsync(descriptor)
