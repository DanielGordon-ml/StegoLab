"""Run the research commands that write benchmark and audit evidence files."""

import json
import logging
import os
import resource
import signal
import sys
import time
from pathlib import Path
from types import FrameType

from pydantic import ValidationError

from backend_service.event_logging import (
    capture_module_events,
    close_run_logger,
    create_run_logger,
    run_directory,
)
from backend_service.failures import ApplicationFailure
from schemas.base import StrictRecord
from schemas.benchmark_identities import BenchmarkFreezeRequest
from schemas.near_duplicate_audit import (
    NearDuplicateAuditReport,
    NearDuplicateAuditRequest,
)

RESEARCH_COMMANDS = ("freeze_benchmark", "validate_benchmark", "audit_near_duplicates")
RESEARCH_MODULES = "backend_service"
AUDIT_RUN_NAME = "near_duplicate_audit_run.json"
MAXIMUM_REQUEST_BYTES = 16 * 1024 * 1024
REQUEST_FAILURE_CODE = "research_request"
INVALID_REQUEST_MESSAGE = (
    "The research request is invalid. Check the documented JSON fields."
)
INVALID_REPORT_MESSAGE = (
    "The research command produced an invalid report. Check the run log."
)
STOPPED_MESSAGE = (
    "The research command was {how} before it finished. Check the output "
    "folder before running it again."
)
ResearchRequest = BenchmarkFreezeRequest | NearDuplicateAuditRequest | Path


class ResearchTerminated(Exception):
    """Request normal cleanup after a termination signal."""


def _terminate(signum: int, frame: FrameType | None) -> None:
    """Leave the active command through its ordinary cleanup boundary."""
    raise ResearchTerminated


def _request_failure() -> ApplicationFailure:
    """Describe an unreadable request without echoing any of it."""
    return ApplicationFailure(
        REQUEST_FAILURE_CODE,
        "The research request could not be read. Use valid JSON.",
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


def _load_command_request(command: str, argument: str) -> ResearchRequest:
    """Validate the command input before any work or logging starts."""
    if command == "freeze_benchmark":
        return BenchmarkFreezeRequest.model_validate_json(_load_request(argument))
    if command == "audit_near_duplicates":
        return NearDuplicateAuditRequest.model_validate_json(_load_request(argument))
    if not argument:
        raise _request_failure()
    return Path(argument)


def _validated_request(command: str, argument: str) -> ResearchRequest | None:
    """Parse the command input, explaining a refusal without echoing it."""
    try:
        return _load_command_request(command, argument)
    except ValidationError:
        print(INVALID_REQUEST_MESSAGE, file=sys.stderr)
    except ApplicationFailure as failure:
        print(failure.message, file=sys.stderr)
    return None


def _run_folder(logger: logging.Logger) -> Path:
    """Return the run log folder or explain that run logging is unavailable."""
    directory = run_directory(logger)
    if directory is None:
        raise ApplicationFailure(
            "research_storage", "The run log folder is missing; check log access."
        )
    return directory


def _record_audit_run(
    directory: Path, report: NearDuplicateAuditReport, started: float
) -> None:
    """Save the measured audit run beside its event log, without any paths."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    document = {
        "schema_version": 1,
        "event": "near_duplicate_audit_completed",
        "audit_identifier": report.audit_identifier,
        "dataset_revision": report.dataset_revision,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_process_mebibytes": round(
            peak / (1024**2 if sys.platform == "darwin" else 1024), 2
        ),
        "hashed_images": report.hashed_images,
        "hashed_by_split": report.hashed_by_split,
        "limited": report.limited,
        "pairs_total_at_maximum_threshold": report.pairs_total_at_maximum_threshold,
        "cross_split_totals": {
            str(level.threshold): level.cross_split_total for level in report.thresholds
        },
        "benchmark_totals": {
            str(level.threshold): level.benchmark_total for level in report.thresholds
        },
        "benchmark_status": (
            None if report.benchmark is None else report.benchmark.status
        ),
    }
    with (directory / AUDIT_RUN_NAME).open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _execute(
    request: ResearchRequest,
    annotations_directory: str | None,
    logger: logging.Logger,
) -> StrictRecord:
    """Dispatch one validated request, importing its service only when selected."""
    if isinstance(request, BenchmarkFreezeRequest):
        from backend_service.benchmark_freeze import freeze_benchmark

        return freeze_benchmark(request)
    if isinstance(request, NearDuplicateAuditRequest):
        from backend_service.near_duplicate_audit import audit_near_duplicates

        directory = _run_folder(logger)
        started = time.perf_counter()
        report = audit_near_duplicates(request)
        _record_audit_run(directory, report, started)
        return report
    from backend_service.benchmark_validation import validate_benchmark

    annotations = None if annotations_directory is None else Path(annotations_directory)
    return validate_benchmark(request, annotations)


def execute_research_command(
    command: str, argument: str, annotations_directory: str | None = None
) -> int:
    """Print validated JSON reports and return stable process exit codes."""
    previous = signal.signal(signal.SIGTERM, _terminate)
    logger = None
    try:
        request = _validated_request(command, argument)
        if request is None:
            return 2
        logger = create_run_logger(
            Path(os.environ.get("STEGOLAB_LOG_DIRECTORY", "logs")), label=command
        )
        logger.info("research_command_started")
        with capture_module_events(logger, RESEARCH_MODULES):
            result = _execute(request, annotations_directory, logger)
        print(result.model_dump_json(indent=2))
        logger.info("research_command_completed")
        return 0
    except KeyboardInterrupt:
        print(STOPPED_MESSAGE.format(how="interrupted"), file=sys.stderr)
        return 130
    except ResearchTerminated:
        print(STOPPED_MESSAGE.format(how="terminated"), file=sys.stderr)
        return 143
    except ValidationError:
        print(INVALID_REPORT_MESSAGE, file=sys.stderr)
        return 1
    except ApplicationFailure as failure:
        print(failure.message, file=sys.stderr)
        return 1
    except (OSError, ValueError, MemoryError):
        print(
            "The research command failed. Check file access, memory, and free space.",
            file=sys.stderr,
        )
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
        if logger is not None:
            close_run_logger(logger)
