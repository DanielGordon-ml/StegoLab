"""Spawn bounded existing services and read their non-secret progress records."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from backend_service.failures import ApplicationFailure
from schemas.base import StrictRecord
from schemas.datasets import DatasetSummary
from schemas.evaluation import EvaluationReport
from schemas.model_exports import ModelExportSummary
from schemas.training import TrainingRun, TrainingStep


def latest_progress(
    root: Path, experiment: str, before: set[Path]
) -> TrainingStep | None:
    """Read the final complete scalar record from a newly created invocation."""
    paths = set((root / "logs").glob(f"*_{experiment}_*/steps.jsonl")) - before
    latest = None
    for path in paths:
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            continue
        try:
            lines = path.read_bytes().splitlines()
            if lines:
                candidate = TrainingStep.model_validate_json(lines[-1])
                if latest is None or candidate.global_step > latest.global_step:
                    latest = candidate
        except (OSError, ValueError, ValidationError):
            continue
    return latest


def execute_command(
    root: Path,
    command: str,
    document: dict[str, object],
    *,
    on_process: Callable[[subprocess.Popen[bytes] | None], None],
    on_progress: Callable[[TrainingStep], None],
    stop_requested: Callable[[], bool],
) -> tuple[int, dict[str, object]]:
    """Run one command without a shell, keeping private paths out of public errors."""
    experiment = str(document.get("experiment_identifier", ""))
    before = set((root / "logs").glob(f"*_{experiment}_*/steps.jsonl"))
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH", "")))
    )
    environment["STEGOLAB_LOG_DIRECTORY"] = str(root / "logs")
    environment["STEGOLAB_WORKER_PARENT_IDENTIFIER"] = str(os.getpid())
    output_file = tempfile.TemporaryFile()
    error_file = tempfile.TemporaryFile()
    process = subprocess.Popen(
        [sys.executable, "-m", "backend_service.workflow_worker", command, "-"],
        stdin=subprocess.PIPE,
        stdout=output_file,
        stderr=error_file,
        cwd=root,
        env=environment,
        start_new_session=True,
    )
    on_process(process)
    sent = False
    previous_step = -1
    started = time.monotonic()
    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps(document).encode())
        process.stdin.close()
        while process.poll() is None:
            if (
                os.fstat(output_file.fileno()).st_size > 2 * 1024 * 1024
                or os.fstat(error_file.fileno()).st_size > 65536
                or time.monotonic() - started > 7500
            ):
                raise ValueError("The bounded worker allowance was exceeded.")
            progress = (
                latest_progress(root, experiment, before)
                if command == "train"
                else None
            )
            if progress is not None:
                if progress.global_step != previous_step:
                    on_progress(progress)
                    previous_step = progress.global_step
                if stop_requested() and not sent:
                    # Published progress proves the safe-save handler is installed.
                    process.send_signal(signal.SIGTERM)
                    sent = True
            time.sleep(0.1)
        output_file.seek(0)
        output = output_file.read(2 * 1024 * 1024 + 1)
        return process.returncode, validate_result(command, output)
    except (OSError, ValueError, ValidationError):
        raise ApplicationFailure(
            "workflow_failed",
            "The operation could not finish. Check the dataset, checkpoint, "
            "remaining time, and storage.",
        ) from None
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
    if len(output) > 2 * 1024 * 1024:
        raise ValueError("Result exceeds the bounded metadata size.")
    models: dict[str, type[StrictRecord]] = {
        "train": TrainingRun,
        "evaluate": EvaluationReport,
        "export_models": ModelExportSummary,
        "prepare_dataset": DatasetSummary,
    }
    return models[command].model_validate_json(output).model_dump(mode="json")


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
