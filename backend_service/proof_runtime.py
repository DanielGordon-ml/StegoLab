"""Bounded local runtime helpers shared by experimental model commands."""

import hashlib
import json
import os
import platform
import random
import signal
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch

from backend_service.failures import ApplicationFailure
from schemas.base import StrictRecord
from schemas.training import TrainingConfiguration


def configure_cpu(configuration: TrainingConfiguration) -> None:
    """Use CPU only and record repeatable numerical settings for continuation."""
    torch.set_num_threads(configuration.cpu_threads)
    torch.use_deterministic_algorithms(True)
    random.seed(configuration.seed)
    np.random.seed(configuration.seed)
    torch.manual_seed(configuration.seed)


def environment_identity(configuration: TrainingConfiguration) -> str:
    """Bind exact continuation to the supported recorded numerical environment."""
    return json.dumps(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "threads": configuration.cpu_threads,
            "deterministic": True,
        },
        sort_keys=True,
    )


def training_source_identity() -> str:
    """Bind continuation to the model, data, loss, and resolved settings code."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in (
        "backend_service/cpu_training.py",
        "backend_service/model_networks.py",
        "backend_service/model_data.py",
        "schemas/training.py",
    ):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def write_record(path: Path, record: StrictRecord) -> None:
    """Atomically replace a small verified JSON report on its own filesystem."""
    data = record.model_dump_json(indent=2).encode("utf-8")
    type(record).model_validate_json(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".proof-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_record(path: Path, record: StrictRecord) -> None:
    """Append one validated bounded progress record without secret inputs."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(record.model_dump_json() + "\n")
        stream.flush()


def run_directory(root: Path, identifier: str) -> Path:
    """Keep each invocation's diagnostics in a distinct date-labelled folder."""
    prefix = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_") + identifier + "_"
    directory = root / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=directory))


def check_output_root(root: Path, dataset: Path | None = None) -> None:
    """Keep model outputs separate from immutable prepared dataset contents."""
    if dataset is not None and (root == dataset or root.is_relative_to(dataset)):
        raise ApplicationFailure(
            "proof_output", "Choose an output folder outside the prepared dataset.", 422
        )
    root.mkdir(parents=True, exist_ok=True)


class StopRequest:
    """Request a save after a complete optimizer step instead of interrupting it."""

    def __init__(self) -> None:
        """Prepare a non-throwing first-signal handler."""
        self.signum: int | None = None
        self.previous: dict[int, Any] = {}

    def __enter__(self) -> "StopRequest":
        """Install temporary command-scoped interruption handlers."""
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous[number] = signal.signal(number, self._receive)
        return self

    def _receive(self, signum: int, frame: FrameType | None) -> None:
        """Delay interruption until the trainer reaches a checkpoint boundary."""
        self.signum = signum

    def __exit__(self, *arguments: object) -> None:
        """Restore the caller's signal handlers after the safe save."""
        for number, handler in self.previous.items():
            signal.signal(number, handler)
