"""Safe experimental CPU commands without enabling application model routes."""

import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from backend_service.failures import ApplicationFailure
from schemas.base import StrictRecord
from schemas.evaluation import EvaluationReport
from schemas.pilot_preflight import PilotPreflightReport
from schemas.pilot_training import PilotEvaluationReport, PilotTrainingRun
from schemas.training import (
    EvaluationRequest,
    ExportRequest,
    TrainingRequest,
    TrainingRun,
)

MODEL_COMMANDS = (
    "train",
    "evaluate",
    "export_models",
    "inspect_checkpoint",
    "preflight_pilot",
)


def _request_text(argument: str) -> str:
    """Bound JSON requests and avoid echoing invalid input or private paths."""
    if argument == "-":
        data = sys.stdin.buffer.read(65537)
    else:
        with Path(argument).open("rb") as stream:
            data = stream.read(65537)
    if len(data) > 65536:
        raise ValueError("Request is too large.")
    return data.decode("utf-8")


def _dispatch(command: str, argument: str) -> StrictRecord:
    """Load model dependencies only for explicitly selected model commands."""
    if command == "inspect_checkpoint":
        from backend_service.training_checkpoints import inspect_checkpoint

        return inspect_checkpoint(Path(argument))
    document = _request_text(argument)
    value = json.loads(document)
    if command == "preflight_pilot" or (
        isinstance(value, dict) and value.get("schema_version") == 2
    ):
        from backend_service.pilot_commands import dispatch_pilot

        return dispatch_pilot(command, document)
    if command == "train":
        from backend_service.cpu_training import train

        return train(TrainingRequest.model_validate_json(document))
    if command == "evaluate":
        from backend_service.proof_operations import evaluate_checkpoint

        return evaluate_checkpoint(EvaluationRequest.model_validate_json(document))
    from backend_service.proof_operations import export_checkpoint

    return export_checkpoint(ExportRequest.model_validate_json(document))


def execute_model_command(command: str, argument: str) -> int:
    """Return fixed safe diagnostics and structured experimental results."""
    from backend_service.event_logging import close_run_logger, create_run_logger

    logger = None
    try:
        logger = create_run_logger(
            Path(os.environ.get("STEGOLAB_LOG_DIRECTORY", "logs"))
        )
        logger.info("experimental_model_command_started")
        result = _dispatch(command, argument)
        print(result.model_dump_json(indent=2))
        if (
            isinstance(result, (EvaluationReport, PilotEvaluationReport))
            and not result.completed
        ):
            logger.info("experimental_model_evaluation_incomplete")
            return 1
        if isinstance(result, PilotPreflightReport) and not result.preparation_passed:
            return 1
        logger.info("experimental_model_command_completed")
        if (
            isinstance(result, (TrainingRun, PilotTrainingRun))
            and result.stop_signal is not None
        ):
            return 128 + result.stop_signal
        return 0
    except ValidationError:
        print(
            "The model request is invalid. Check the documented JSON fields.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print(
            "The operation was interrupted. Previous completed files remain available.",
            file=sys.stderr,
        )
        return 130
    except ApplicationFailure as failure:
        print(failure.message, file=sys.stderr)
        return 1
    except Exception:
        print(
            "The model operation failed. Check compatible files, CPU memory, "
            "and storage.",
            file=sys.stderr,
        )
        return 1
    finally:
        if logger is not None:
            close_run_logger(logger)
