"""Verified pilot checkpoint reads and guarded tuning evaluation."""

from pathlib import Path
from typing import Any, Literal

from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.model_networks import build_models, model_pair_identifier
from backend_service.pilot_data import load_pilot_data
from backend_service.pilot_evaluation import evaluate_pilot_models
from backend_service.pilot_execution import pilot_execution
from backend_service.pilot_runtime import check_pilot_deadline, configure_pilot
from backend_service.proof_runtime import check_output_root, run_directory, write_record
from backend_service.training_checkpoints import read_checkpoint_state
from schemas.pilot_training import (
    PilotEvaluationReport,
    PilotEvaluationRequest,
    PilotTrainingConfiguration,
)


def frozen_pilot_models(
    checkpoint: Path,
    experiment_identifier: str,
    device: Literal["cpu", "cuda"] = "cpu",
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Read verified pilot weights without restoring training random state."""
    state = read_checkpoint_state(checkpoint)
    if state["identities"].get("experiment_identifier") != experiment_identifier:
        raise ApplicationFailure(
            "experiment_mismatch",
            "Use the checkpoint's original experiment identifier.",
            422,
        )
    configuration = PilotTrainingConfiguration.model_validate(state["configuration"])
    runtime_configuration = configuration.model_copy(update={"device": device})
    selected = configure_pilot(runtime_configuration)
    encoder, decoder = build_models(configuration.seed)
    encoder.load_state_dict(state["encoder"], strict=True)
    decoder.load_state_dict(state["decoder"], strict=True)
    encoder.to(selected).eval()
    decoder.to(selected).eval()
    return encoder, decoder, state


def evaluate_pilot_checkpoint(request: PilotEvaluationRequest) -> PilotEvaluationReport:
    """Record every attempted tuning recovery inside its selected budget context."""
    request = PilotEvaluationRequest.model_validate(request)
    root = Path(request.output_root).resolve()
    check_output_root(root, Path(request.dataset_directory).resolve())
    with pilot_execution(request) as deadline:
        check_pilot_deadline(deadline)
        encoder, decoder, state = frozen_pilot_models(
            Path(request.checkpoint),
            request.experiment_identifier,
            request.device,
        )
        data = load_pilot_data(Path(request.dataset_directory), deadline=deadline)
        if data.manifest.revision != state["identities"].get(
            "dataset_revision"
        ) or data.selection.selection_checksum != state["identities"].get("selection"):
            raise ApplicationFailure(
                "dataset_mismatch",
                "Use the dataset revision and selection recorded in the checkpoint.",
                422,
            )
        report = evaluate_pilot_models(
            encoder,
            decoder,
            data,
            model_pair_identifier(encoder, decoder),
            device=next(encoder.parameters()).device,
            deadline=deadline,
            smoke=request.execution_mode == "cpu_smoke",
        )
        write_record(
            run_directory(root, request.experiment_identifier) / "evaluation.json",
            report,
        )
        return report
