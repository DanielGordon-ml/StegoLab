"""Run local dataset commands with safe diagnostics and interruption cleanup."""

import os
import signal
import sys
from pathlib import Path
from types import FrameType

from pydantic import ValidationError

from backend_service.event_logging import close_run_logger, create_run_logger
from backend_service.failures import ApplicationFailure

DATASET_COMMANDS = ("prepare_dataset", "inspect_dataset", "validate_dataset")


class DatasetTerminated(Exception):
    """Request normal context cleanup after a termination signal."""


def _terminate(signum: int, frame: FrameType | None) -> None:
    """Leave the active import through its ordinary cleanup boundary."""
    raise DatasetTerminated


def _load_request(argument: str) -> str:
    """Read a bounded request document from a file or standard input."""
    maximum_bytes = 16 * 1024 * 1024
    try:
        if argument == "-":
            data = sys.stdin.buffer.read(maximum_bytes + 1)
        else:
            with Path(argument).open("rb") as stream:
                data = stream.read(maximum_bytes + 1)
        if len(data) > maximum_bytes:
            raise ValueError
        return data.decode("utf-8")
    except (OSError, ValueError):
        raise ApplicationFailure(
            "dataset_request", "The dataset request could not be read. Use valid JSON."
        ) from None


def execute_dataset_command(command: str, argument: str) -> int:
    """Print validated JSON summaries and return stable process exit codes."""
    from backend_service.dataset_preparation import prepare_dataset
    from backend_service.dataset_validation import inspect_dataset, validate_dataset
    from schemas.datasets import DatasetPreparationRequest

    previous = signal.signal(signal.SIGTERM, _terminate)
    logger = None
    try:
        logger = create_run_logger(
            Path(os.environ.get("STEGOLAB_LOG_DIRECTORY", "logs"))
        )
        logger.info("dataset_command_started")
        if command == "prepare_dataset":
            request = DatasetPreparationRequest.model_validate_json(
                _load_request(argument)
            )
            result = prepare_dataset(request, progress=logger.info)
        elif command == "inspect_dataset":
            result = inspect_dataset(Path(argument))
        else:
            result = validate_dataset(Path(argument))
        print(result.model_dump_json(indent=2))
        logger.info("dataset_command_completed")
        return 0
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
