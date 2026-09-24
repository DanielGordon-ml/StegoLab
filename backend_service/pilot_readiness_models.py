"""Optional learned-weight export and authentic PNG checks on a later GPU host."""

from pathlib import Path

import torch

from backend_service.failures import ApplicationFailure
from backend_service.model_export_verification import verify_exports
from backend_service.model_exports import export_models
from backend_service.model_networks import model_pair_identifier
from backend_service.pilot_data import load_pilot_data
from backend_service.pilot_evaluation import evaluate_pilot_models
from backend_service.pilot_export import cuda_runtime_dependencies
from backend_service.pilot_operations import frozen_pilot_models
from backend_service.proof_operations import frozen_models
from backend_service.proof_runtime import write_record
from backend_service.training_checkpoints import read_checkpoint_state
from schemas.model_exports import ExportMetadata
from schemas.pilot_readiness import GpuReadinessRequest


def verify_learned_cuda_exports(
    request: GpuReadinessRequest, directory: Path, deadline: float
) -> list[str]:
    """Create isolated version-two packages and compare CUDA graphs to eager models."""
    encoder, decoder = _learned_models(request)
    metadata = ExportMetadata(
        compatibility_identifier=model_pair_identifier(encoder, decoder),
        source_identifier="gpu_readiness_frozen_learned_weights",
    )
    directory.mkdir(parents=True, exist_ok=True)
    exports = export_models(
        encoder,
        decoder,
        directory / "packages",
        metadata,
        deadline=deadline,
        runtime_dependencies={"cuda": cuda_runtime_dependencies()},
    )
    report = verify_exports(
        encoder, decoder, Path(exports.directory), deadline=deadline, device="cuda"
    )
    destination = directory / "cuda_verification.json"
    write_record(destination, report)
    return [str(destination), exports.directory]


def verify_learned_cuda_recovery(
    request: GpuReadinessRequest, directory: Path, deadline: float
) -> list[str]:
    """Attempt every frozen tuning cover once with full authenticated messages."""
    encoder, decoder = _learned_models(request)
    selected = torch.device("cuda", 0)
    encoder.to(selected).eval()
    decoder.to(selected).eval()
    data = load_pilot_data(Path(request.dataset_directory), deadline=deadline)
    report = evaluate_pilot_models(
        encoder,
        decoder,
        data,
        model_pair_identifier(encoder, decoder),
        device=selected,
        deadline=deadline,
    )
    destination = directory / "cuda_png_evaluation.json"
    write_record(destination, report)
    if not report.completed or report.exact_recovery_count != 32:
        raise ApplicationFailure(
            "pilot_recovery_failed",
            "The frozen learned weights did not recover all 32 full PNG messages. "
            "Keep the recorded failures; no threshold was lowered or retried.",
            422,
        )
    return [str(destination)]


def _learned_models(
    request: GpuReadinessRequest,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load verified legacy proof or pilot weights on CPU without resuming them."""
    if (
        request.learned_checkpoint is None
        or request.learned_experiment_identifier is None
    ):
        raise ApplicationFailure(
            "pilot_learned_checkpoint",
            "Provide the frozen learned checkpoint identity.",
        )
    checkpoint = Path(request.learned_checkpoint)
    state = read_checkpoint_state(checkpoint)
    architecture = state["configuration"].get("architecture")
    if architecture == "dense_pilot_v1":
        encoder, decoder, _ = frozen_pilot_models(
            checkpoint, request.learned_experiment_identifier
        )
    elif architecture == "dense_cpu_v1":
        encoder, decoder, _ = frozen_models(
            checkpoint, request.learned_experiment_identifier
        )
    else:
        raise ApplicationFailure(
            "pilot_checkpoint_architecture",
            "Use a supported frozen CPU-proof or pilot checkpoint.",
            422,
        )
    return encoder, decoder
