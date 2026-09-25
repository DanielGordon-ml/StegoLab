"""Explicit devices, reproducible numerical settings, and pilot provenance."""

import fcntl
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

from backend_service.failures import ApplicationFailure
from backend_service.model_export_devices import configure_cuda_numerics
from schemas.pilot_training import PilotTrainingConfiguration


def require_device(device: Literal["cpu", "cuda"]) -> torch.device:
    """Reject unavailable CUDA instead of silently changing the requested device."""
    if device not in ("cpu", "cuda"):
        raise ApplicationFailure(
            "pilot_device_invalid", "Choose CPU or CUDA explicitly.", 422
        )
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ApplicationFailure(
                "pilot_cuda_unavailable",
                "CUDA requires exactly one visible supported GPU. Run the GPU "
                "readiness check; CPU fallback is disabled.",
                422,
            )
        return torch.device("cuda", 0)
    return torch.device("cpu")


def configure_pilot(configuration: PilotTrainingConfiguration) -> torch.device:
    """Initialize one recorded deterministic FP32 environment before models."""
    if configuration.device == "cuda":
        configure_cuda_numerics()
    device = require_device(configuration.device)
    torch.set_num_threads(configuration.cpu_threads)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        cast(Callable[[], None], torch.cuda.init)()
    random.seed(configuration.seed)
    np.random.seed(configuration.seed % 2**32)
    torch.manual_seed(configuration.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(configuration.seed)
    return device


def pilot_environment(configuration: PilotTrainingConfiguration) -> str:
    """Bind continuation to versions, hardware and numerical behavior."""
    values: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": configuration.device,
        "threads": configuration.cpu_threads,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "precision": configuration.precision,
    }
    if configuration.device == "cuda":
        values.update(
            cuda=torch.version.cuda,
            cudnn=cast(Callable[[], int | None], torch.backends.cudnn.version)(),
            device_name=torch.cuda.get_device_name(0),
            device_capability=list(torch.cuda.get_device_capability(0)),
            device_count=torch.cuda.device_count(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            driver=_driver_version(),
        )
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _driver_version() -> str:
    """Read the installed driver only in an explicitly initialized GPU environment."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        versions = set(result.stdout.strip().splitlines())
        if len(versions) != 1:
            raise ValueError
        version = versions.pop().strip()
        if not version or any(character not in "0123456789." for character in version):
            raise ValueError
        return version
    except (OSError, ValueError, subprocess.SubprocessError):
        raise ApplicationFailure(
            "pilot_driver_unavailable",
            "The NVIDIA driver version could not be "
            "verified. Restore nvidia-smi before starting or resuming GPU training.",
            422,
        ) from None


@contextmanager
def pilot_experiment_lock(root: Path, identifier: str) -> Iterator[None]:
    """Exclude concurrent writers to the same experiment in both execution modes."""
    directory = root / "state" / "pilot_operations"
    descriptor: int | None = None
    try:
        if directory.parent.is_symlink() or directory.is_symlink():
            raise OSError
        directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            directory / f"{identifier}.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise ApplicationFailure(
            "pilot_experiment_busy",
            "This experiment is already running or its "
            "lock is unavailable. Wait for it to finish before resuming.",
        ) from None
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def pilot_source_identity() -> str:
    """Freeze the implementation affecting learning, sampling and continuation."""
    root = Path(__file__).resolve().parents[1]
    names = {
        "backend_service/model_networks.py",
        "backend_service/model_critic.py",
        "backend_service/model_export_devices.py",
        "backend_service/pilot_runtime.py",
        "backend_service/pilot_data.py",
        "backend_service/pilot_sampler.py",
        "backend_service/training_checkpoint_state.py",
        "backend_service/training_checkpoints.py",
        "backend_service/pilot_evaluation.py",
        "backend_service/pilot_evaluation_images.py",
        "schemas/pilot_training.py",
    }
    for directory, pattern in (
        ("backend_service", "pilot_training*.py"),
        ("schemas", "pilot_data*.py"),
    ):
        names.update(
            path.relative_to(root).as_posix()
            for path in (root / directory).glob(pattern)
        )
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def check_pilot_deadline(deadline: float) -> None:
    """Stop between bounded operations when the caller's allowance expires."""
    if time.monotonic() >= deadline:
        raise ApplicationFailure(
            "pilot_deadline",
            "The pilot operation reached its time limit. "
            "Preserve the last completed checkpoint and report incomplete checks.",
        )


def synchronize_device(device: torch.device) -> None:
    """Finish queued GPU operations before recording timing or moving outputs."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
