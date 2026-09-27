"""Run local dataset commands with safe diagnostics and interruption cleanup."""

import json
import logging
import os
import resource
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

from pydantic import ValidationError

from backend_service.event_logging import (
    capture_module_events,
    close_run_logger,
    create_run_logger,
    run_directory,
)
from backend_service.failures import ApplicationFailure
from schemas.base import StrictRecord
from schemas.dataset_fetch import DatasetFetchDocument, DatasetFetchSummary
from schemas.dataset_records import DatasetPreparationRequest
from schemas.dataset_sources import DatasetInspectionRequest

if TYPE_CHECKING:
    from backend_service.dataset_sources.upload_sessions import DatasetUploadStore

DATASET_COMMANDS = (
    "prepare_dataset",
    "inspect_dataset",
    "validate_dataset",
    "fetch_dataset",
    "inspect_source",
)
SOURCE_COMMANDS = ("fetch_dataset", "inspect_source")
SOURCE_MODULES = "backend_service.dataset_sources"
DEFAULT_DATA_ROOT = "data"
DEFAULT_CACHE_ROOT = ".cache/stegolab/datasets"
DEFAULT_STATE_DIRECTORY = ".runtime"
FETCH_RUN_NAME = "dataset_fetch_run.json"
MAXIMUM_REQUEST_BYTES = 16 * 1024 * 1024


class DatasetTerminated(Exception):
    """Request normal context cleanup after a termination signal."""


@dataclass(frozen=True)
class SourceInspectionCommand:
    """An inspection request with the workspace roots it should look at."""

    request: DatasetInspectionRequest
    data_root: Path
    cache_root: Path


CommandRequest = (
    DatasetPreparationRequest | DatasetFetchDocument | SourceInspectionCommand | Path
)


def _terminate(signum: int, frame: FrameType | None) -> None:
    """Leave the active import through its ordinary cleanup boundary."""
    raise DatasetTerminated


def _request_failure() -> ApplicationFailure:
    """Describe an unreadable request without echoing any of it."""
    return ApplicationFailure(
        "dataset_request", "The dataset request could not be read. Use valid JSON."
    )


def _load_request(argument: str) -> str:
    """Read a bounded request document from a file or standard input."""
    try:
        if argument == "-":
            data = sys.stdin.buffer.read(MAXIMUM_REQUEST_BYTES + 1)
        else:
            with Path(argument).open("rb") as stream:
                data = stream.read(MAXIMUM_REQUEST_BYTES + 1)
        if len(data) > MAXIMUM_REQUEST_BYTES:
            raise ValueError
        return data.decode("utf-8")
    except (OSError, ValueError):
        raise _request_failure() from None


def _load_inspection(argument: str) -> SourceInspectionCommand:
    """Parse an inspection request plus optional workspace root overrides."""
    try:
        document = json.loads(_load_request(argument))
    except ValueError:
        raise _request_failure() from None
    if not isinstance(document, dict):
        raise _request_failure()
    fields = dict(document)
    roots = {name: fields.pop(name, None) for name in ("data_root", "cache_root")}
    if any(
        value is not None and not isinstance(value, str) for value in roots.values()
    ):
        raise _request_failure()
    workspace = Path(os.environ.get("STEGOLAB_WORKSPACE_ROOT", "."))
    defaults = {"data_root": DEFAULT_DATA_ROOT, "cache_root": DEFAULT_CACHE_ROOT}
    chosen = {
        name: Path(value) if isinstance(value, str) else workspace / defaults[name]
        for name, value in roots.items()
    }
    request = DatasetInspectionRequest.model_validate(fields)
    return SourceInspectionCommand(request, chosen["data_root"], chosen["cache_root"])


def _load_command_request(command: str, argument: str) -> CommandRequest:
    """Validate the command input before any work or logging starts."""
    if command == "prepare_dataset":
        return DatasetPreparationRequest.model_validate_json(_load_request(argument))
    if command == "fetch_dataset":
        return DatasetFetchDocument.model_validate_json(_load_request(argument))
    if command == "inspect_source":
        return _load_inspection(argument)
    return Path(argument)


def _record_fetch_run(
    directory: Path, summary: DatasetFetchSummary, started: float
) -> None:
    """Save measured fetch metadata beside the run's event log."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "schema_version": 1,
        "event": f"dataset_fetch_{summary.status}",
        "summary": summary.model_dump(mode="json"),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_process_mebibytes": round(
            peak / (1024**2 if sys.platform == "darwin" else 1024), 2
        ),
    }
    with (directory / FETCH_RUN_NAME).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _upload_store() -> "DatasetUploadStore":
    """Open the upload store under the same state directory the web service uses.

    Nothing is created until an uploaded archive is actually read, so sources
    of other kinds never touch the state directory.
    """
    from backend_service.dataset_sources.upload_sessions import DatasetUploadStore

    state = os.environ.get("STEGOLAB_DATA_DIRECTORY", DEFAULT_STATE_DIRECTORY)
    return DatasetUploadStore(Path(state))


def _execute_fetch(
    document: DatasetFetchDocument, logger: logging.Logger
) -> tuple[StrictRecord, int]:
    """Run a fetch with progress beside the run log; map a stop to its signal."""
    from backend_service.dataset_sources.fetch import fetch_dataset
    from backend_service.dataset_sources.progress import FetchProgressSink

    directory = run_directory(logger)
    if directory is None:
        raise ApplicationFailure(
            "dataset_storage", "The run log folder is missing; check log access."
        )
    started = time.perf_counter()
    summary = fetch_dataset(
        document, progress=FetchProgressSink(directory), uploads=_upload_store()
    )
    _record_fetch_run(directory, summary, started)
    if summary.stop_signal is not None:
        return summary, 128 + summary.stop_signal
    return summary, 0


def _execute(
    command: str, request: CommandRequest, logger: logging.Logger
) -> tuple[StrictRecord, int]:
    """Dispatch one validated request and return its result and exit code."""
    if isinstance(request, DatasetFetchDocument):
        return _execute_fetch(request, logger)
    if isinstance(request, SourceInspectionCommand):
        from backend_service.dataset_sources.fetch import inspect_source

        inspection = inspect_source(
            request.request.source,
            source_name=request.request.source_name,
            data_root=request.data_root,
            cache_root=request.cache_root,
            uploads=_upload_store(),
            maximum_images=request.request.maximum_images,
        )
        return inspection, 0
    from backend_service.dataset_preparation import prepare_dataset
    from backend_service.dataset_validation import inspect_dataset, validate_dataset

    if isinstance(request, DatasetPreparationRequest):
        return prepare_dataset(request, progress=logger.info), 0
    if command == "inspect_dataset":
        return inspect_dataset(request), 0
    return validate_dataset(request), 0


def execute_dataset_command(command: str, argument: str) -> int:
    """Print validated JSON summaries and return stable process exit codes."""
    previous = signal.signal(signal.SIGTERM, _terminate)
    logger = None
    try:
        request = _load_command_request(command, argument)
        label = request.run_label if isinstance(request, DatasetFetchDocument) else None
        logger = create_run_logger(
            Path(os.environ.get("STEGOLAB_LOG_DIRECTORY", "logs")), label=label
        )
        logger.info("dataset_command_started")
        if command in SOURCE_COMMANDS:
            with capture_module_events(logger, SOURCE_MODULES):
                result, exit_code = _execute(command, request, logger)
        else:
            result, exit_code = _execute(command, request, logger)
        print(result.model_dump_json(indent=2))
        logger.info("dataset_command_completed")
        return exit_code
    except KeyboardInterrupt:
        print(
            "Dataset preparation was interrupted; owned staging was cleaned.",
            file=sys.stderr,
        )
        return 130
    except DatasetTerminated:
        print(
            "Dataset preparation was terminated; owned staging was cleaned.",
            file=sys.stderr,
        )
        return 143
    except ValidationError:
        print(
            "The dataset request is invalid. Check the documented JSON fields.",
            file=sys.stderr,
        )
        return 2
    except ApplicationFailure as failure:
        print(failure.message, file=sys.stderr)
        return 2 if failure.code == "dataset_request" else 1
    except (OSError, ValueError, MemoryError):
        print(
            "Dataset processing failed. Check file access, memory, and free space.",
            file=sys.stderr,
        )
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
        if logger is not None:
            close_run_logger(logger)
