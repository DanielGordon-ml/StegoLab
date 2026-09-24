"""Evaluate or export frozen model pairs against the persistent proof budget."""

import time
from pathlib import Path
from typing import Any

from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.model_data import load_proof_data
from backend_service.model_evaluation import evaluate_models
from backend_service.model_exports import export_models
from backend_service.model_networks import build_models, model_pair_identifier
from backend_service.proof_runtime import (
    check_output_root,
    configure_cpu,
    run_directory,
    write_record,
)
from backend_service.training_budget import ProofBudget
from backend_service.training_checkpoints import read_checkpoint_state
from schemas.evaluation import EvaluationReport
from schemas.model_exports import ExportMetadata, ModelExportSummary
from schemas.training import EvaluationRequest, ExportRequest, TrainingConfiguration


def frozen_models(
    checkpoint: Path, experiment_identifier: str
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Load only verified CPU weights without resuming training random state."""
    state = read_checkpoint_state(checkpoint)
    if state["identities"].get("experiment_identifier") != experiment_identifier:
        raise ApplicationFailure(
            "experiment_mismatch",
            "Use the checkpoint's original experiment identifier.",
        )
    configuration = TrainingConfiguration.model_validate(state["configuration"])
    configure_cpu(configuration)
    encoder, decoder = build_models(configuration.seed)
    encoder.load_state_dict(state["encoder"], strict=True)
    decoder.load_state_dict(state["decoder"], strict=True)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, state


def evaluate_checkpoint(request: EvaluationRequest) -> EvaluationReport:
    """Save every attempted tuning result even when the learning gate fails."""
    root = Path(request.output_root).resolve()
    check_output_root(root, Path(request.dataset_directory).resolve())
    with ProofBudget(
        root / "state" / "cpu_proof", request.experiment_identifier, resume=True
    ) as budget:
        deadline = time.monotonic() + budget.remaining_seconds
        encoder, decoder, state = frozen_models(
            Path(request.checkpoint), request.experiment_identifier
        )
        data = load_proof_data(Path(request.dataset_directory), deadline=deadline)
        if data.manifest.revision != state["identities"]["dataset_revision"]:
            raise ApplicationFailure(
                "dataset_mismatch",
                "Use the dataset revision recorded in the checkpoint.",
            )
        report = evaluate_models(
            encoder,
            decoder,
            data,
            model_pair_identifier(encoder, decoder),
            deadline=deadline,
            lightweight=request.lightweight,
        )
        logs = run_directory(root, request.experiment_identifier)
        write_record(logs / "evaluation.json", report)
        return report


def export_checkpoint(request: ExportRequest) -> ModelExportSummary:
    """Publish separate experimental packages without promoting model capacity."""
    root = Path(request.output_root).resolve()
    check_output_root(root)
    with ProofBudget(
        root / "state" / "cpu_proof", request.experiment_identifier, resume=True
    ) as budget:
        deadline = time.monotonic() + budget.remaining_seconds
        encoder, decoder, state = frozen_models(
            Path(request.checkpoint), request.experiment_identifier
        )
        metadata = ExportMetadata(
            compatibility_identifier=model_pair_identifier(encoder, decoder),
            source_identifier=f"{request.experiment_identifier}_step_{state['global_step']}",
        )
        (root / "models").mkdir(parents=True, exist_ok=True)
        result = export_models(
            encoder,
            decoder,
            root / "models" / request.export_name,
            metadata,
            deadline=deadline,
        )
        logs = run_directory(root, request.experiment_identifier)
        write_record(logs / "exports.json", result)
        return result
