"""Spawn bounded existing services and read their non-secret progress records."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from backend_service.failures import ApplicationFailure
from schemas.base import StrictRecord
from schemas.dataset_fetch import DatasetFetchProgress, DatasetFetchSummary
from schemas.datasets import DatasetSummary
from schemas.evaluation import EvaluationReport
from schemas.model_exports import ModelExportSummary
from schemas.training import TrainingRun, TrainingStep

DEFAULT_TIME_LIMIT_SECONDS = 7_500
TIME_LIMITS = {"fetch_dataset": 86_400}
STALL_LIMITS = {"fetch_dataset": 1_800}
# Only the phases that talk to the network report progress continuously; the
# inventory and integrity passes of later phases can stay silent for a long
# time while working correctly, so the stall rule leaves them alone.
STALL_PHASES = frozenset({"resolving", "downloading"})
MAXIMUM_PROGRESS_BYTES = 1024 * 1024
MAXIMUM_OUTPUT_BYTES = 2 * 1024 * 1024
MAXIMUM_ERROR_BYTES = 65536
POLL_INTERVAL_SECONDS = 0.1
RESULT_MODELS: dict[str, type[StrictRecord]] = {
    "train": TrainingRun,
    "evaluate": EvaluationReport,
    "export_models": ModelExportSummary,
    "prepare_dataset": DatasetSummary,
    "fetch_dataset": DatasetFetchSummary,
}
_MESSAGES = {
    "workflow_failed": (
        "The operation could not finish. Check the dataset, checkpoint, "
        "remaining time, and storage."
    ),
    "workflow_stalled": (
        "The download made no progress for 30 minutes and was stopped."
    ),
}


def execution_failure(code: str) -> ApplicationFailure:
    """Return the fixed plain message for a worker execution failure code."""
    return ApplicationFailure(code, _MESSAGES[code])


def read_training_step(path: Path) -> tuple[int, StrictRecord]:
    """Parse the last complete scalar record of a training step log."""
    lines = path.read_bytes().splitlines()
    if not lines:
        raise ValueError("The step log holds no complete record yet.")
    step = TrainingStep.model_validate_json(lines[-1])
    return step.global_step, step


def read_fetch_progress(path: Path) -> tuple[int, StrictRecord]:
    """Parse the whole progress record a fetch rewrites atomically."""
    record = DatasetFetchProgress.model_validate_json(path.read_bytes())
    return record.sequence, record


@dataclass(frozen=True)
class ProgressReader:
    """Locate and parse the progress file one command writes in its run folder."""

    file_name: str
    label_field: str
    parse: Callable[[Path], tuple[int, StrictRecord]]

    def label(self, document: dict[str, object]) -> str:
        """Return the run-folder label the command derives from its document."""
        return str(document.get(self.label_field, ""))

    def candidates(self, root: Path, label: str) -> set[Path]:
        """Find every progress file written for the label under the log folder."""
        return set((root / "logs").glob(f"*_{label}_*/{self.file_name}"))


PROGRESS_READERS = {
    "train": ProgressReader("steps.jsonl", "experiment_identifier", read_training_step),
    "fetch_dataset": ProgressReader(
        "fetch_progress.json", "run_label", read_fetch_progress
    ),
}


def latest_progress(
    root: Path, reader: ProgressReader, label: str, before: set[Path]
) -> tuple[int, StrictRecord] | None:
    """Read the most advanced complete record from a newly created invocation."""
    latest: tuple[int, StrictRecord] | None = None
    for path in reader.candidates(root, label) - before:
        try:
            if path.is_symlink() or path.stat().st_size > MAXIMUM_PROGRESS_BYTES:
                continue
            candidate = reader.parse(path)
        except (OSError, ValueError, ValidationError):
            continue
        if latest is None or candidate[0] > latest[0]:
            latest = candidate
    return latest


def stall_applies(record: StrictRecord | None) -> bool:
    """Say whether silence from the worker counts as a stall right now.

    Before the first record the worker is still starting, which the rule
    covers; afterwards only the network phases of a fetch are watched.
    """
    if record is None:
        return True
    return getattr(record, "phase", None) in STALL_PHASES


def child_environment(root: Path, extra: dict[str, str] | None) -> dict[str, str]:
    """Build the worker environment with the source tree and log folder set."""
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH", "")))
    )
    environment["STEGOLAB_LOG_DIRECTORY"] = str(root / "logs")
    environment["STEGOLAB_WORKER_PARENT_IDENTIFIER"] = str(os.getpid())
    environment.update(extra or {})
    return environment


def execute_command(
    root: Path,
    command: str,
    document: dict[str, object],
    *,
    on_process: Callable[[subprocess.Popen[bytes] | None], None],
    on_progress: Callable[[StrictRecord], None],
    stop_requested: Callable[[], bool],
    environment: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    """Run one command without a shell, keeping private paths out of public errors."""
    reader = PROGRESS_READERS.get(command)
    label = reader.label(document) if reader is not None else ""
    before = reader.candidates(root, label) if reader is not None else set()
    time_limit = TIME_LIMITS.get(command, DEFAULT_TIME_LIMIT_SECONDS)
    stall_limit = STALL_LIMITS.get(command)
    output_file = tempfile.TemporaryFile()
    error_file = tempfile.TemporaryFile()
    process = subprocess.Popen(
        [sys.executable, "-m", "backend_service.workflow_worker", command, "-"],
        stdin=subprocess.PIPE,
        stdout=output_file,
        stderr=error_file,
        cwd=root,
        env=child_environment(root, environment),
        start_new_session=True,
    )
    on_process(process)
    sent = False
    previous_position = -1
    published: StrictRecord | None = None
    started = time.monotonic()
    advanced = started
    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps(document).encode())
        process.stdin.close()
        while process.poll() is None:
            now = time.monotonic()
            if (
                os.fstat(output_file.fileno()).st_size > MAXIMUM_OUTPUT_BYTES
                or os.fstat(error_file.fileno()).st_size > MAXIMUM_ERROR_BYTES
                or now - started > time_limit
            ):
                raise ValueError("The bounded worker allowance was exceeded.")
            latest = (
                latest_progress(root, reader, label, before)
                if reader is not None
                else None
            )
            if latest is not None:
                position, record = latest
                if position != previous_position:
                    on_progress(record)
                    previous_position = position
                    published = record
                    advanced = now
                if stop_requested() and not sent:
                    # Published progress proves the safe-stop handler is installed.
                    process.send_signal(signal.SIGTERM)
                    sent = True
            if (
                stall_limit is not None
                and stall_applies(published)
                and now - advanced > stall_limit
            ):
                raise execution_failure("workflow_stalled")
            time.sleep(POLL_INTERVAL_SECONDS)
        output_file.seek(0)
        output = output_file.read(MAXIMUM_OUTPUT_BYTES + 1)
        return process.returncode, validate_result(command, output)
    except (OSError, ValueError, ValidationError):
        raise execution_failure("workflow_failed") from None
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        on_process(None)
        output_file.close()
        error_file.close()


def validate_result(command: str, output: bytes) -> dict[str, object]:
    """Validate the exact existing command report before exposing any fields."""
    if len(output) > MAXIMUM_OUTPUT_BYTES:
        raise ValueError("Result exceeds the bounded metadata size.")
    return RESULT_MODELS[command].model_validate_json(output).model_dump(mode="json")


def public_result(result: dict[str, object]) -> dict[str, object]:
    """Remove private output locations while retaining measured scientific results."""
    private = {
        "checkpoint",
        "directory",
        "encoder_directory",
        "decoder_directory",
        "report_directory",
    }
    return {key: value for key, value in result.items() if key not in private}
