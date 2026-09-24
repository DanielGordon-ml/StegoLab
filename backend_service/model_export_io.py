"""Bounded files and tensor conversion shared by independent runtimes."""

import hashlib
import importlib.metadata
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from backend_service.failures import ApplicationFailure
from schemas.model_exports import ModelExportManifest

MAXIMUM_EXPORT_FILE = 16 * 1024**2
MAXIMUM_EXPORT_BYTES = 64 * 1024**2


def export_failure() -> ApplicationFailure:
    """Return one safe package error without exposing private contents."""
    return ApplicationFailure(
        "model_export_unavailable",
        "The model package is missing, damaged, or incompatible. "
        "Use a complete matching package and its recorded dependencies.",
        422,
    )


def require_export_space(directory: Path, additional_bytes: int) -> None:
    """Keep ten GiB or ten percent of the disk free after bounded writes."""
    usage = shutil.disk_usage(directory)
    if usage.free - additional_bytes < max(10 * 1024**3, (usage.total + 9) // 10):
        raise ApplicationFailure(
            "model_export_space",
            "Export would exceed the free-space reserve. Free disk space and retry.",
            422,
        )


def read_export_file(path: Path, maximum: int = MAXIMUM_EXPORT_FILE) -> bytes:
    """Read bounded regular bytes without following a final symlink."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        information = os.fstat(stream.fileno())
        if not stat.S_ISREG(information.st_mode) or information.st_size > maximum:
            raise export_failure()
        content = stream.read(maximum + 1)
        if len(content) > maximum or len(content) != information.st_size:
            raise export_failure()
        return content


def verify_package(
    directory: Path, *, device: Literal["cpu", "cuda"] = "cpu"
) -> ModelExportManifest:
    """Check every declared package file before deserializing its graph."""
    if directory.is_symlink() or not directory.is_dir():
        raise export_failure()
    manifest = ModelExportManifest.model_validate_json(
        read_export_file(directory / "manifest.json", 256 * 1024)
    )
    if "model.pt2" not in manifest.files or "runtime.py" not in manifest.files:
        raise export_failure()
    total = 0
    for relative, record in manifest.files.items():
        parts = relative.split("/")
        if any(part in ("", ".", "..") for part in parts) or "\\" in relative:
            raise export_failure()
        candidate = directory
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise export_failure()
        content = read_export_file(candidate)
        total += len(content)
        if (
            total > MAXIMUM_EXPORT_BYTES
            or len(content) != record.size_bytes
            or hashlib.sha256(content).hexdigest() != record.checksum
        ):
            raise export_failure()
    if sys.version_info[:2] != (3, 12):
        raise export_failure()
    if device == "cuda" and manifest.format_version != 2:
        raise export_failure()
    dependencies = (
        manifest.runtime_dependencies[device]
        if manifest.format_version == 2
        else manifest.dependencies
    )
    for package, expected in dependencies.items():
        actual = importlib.metadata.version(package)
        if package == "torch" and device == "cpu":
            actual = actual.split("+", 1)[0]
        if actual != expected:
            raise export_failure()
    return manifest


def array_tensor(values: NDArray[np.float32], channels: int) -> torch.Tensor:
    """Validate a CPU proof tensor before allocating graph activations."""
    if (
        values.dtype != np.float32
        or values.ndim != 4
        or values.shape[:2] != (1, channels)
        or not all(512 <= side <= 1024 for side in values.shape[2:])
        or not np.isfinite(values).all()
    ):
        raise export_failure()
    return torch.from_numpy(np.ascontiguousarray(values))


def tensor_array(values: torch.Tensor) -> NDArray[np.float32]:
    """Return detached CPU float32 values for protocol or file handling."""
    return cast(NDArray[np.float32], values.detach().cpu().numpy())


def read_tensor(path: Path, channels: int) -> torch.Tensor:
    """Load a bounded non-pickle NumPy tensor for an independent process."""
    from io import BytesIO

    values = np.load(BytesIO(read_export_file(path)), allow_pickle=False)
    if not isinstance(values, np.ndarray):
        raise export_failure()
    return array_tensor(values, channels)


def write_tensor(path: Path, values: torch.Tensor) -> None:
    """Write a new tensor output without replacing an existing file."""
    require_export_space(path.parent, values.numel() * values.element_size() + 4096)
    with path.open("xb") as stream:
        np.save(stream, tensor_array(values), allow_pickle=False)
