"""Publish portable independent graphs from frozen pilot weights on CPU."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from backend_service.model_export_devices import export_device
from backend_service.model_exports import export_models
from backend_service.model_networks import model_pair_identifier
from backend_service.pilot_execution import pilot_execution
from backend_service.proof_runtime import check_output_root, run_directory, write_record
from schemas.model_exports import ExportMetadata, ModelExportSummary
from schemas.pilot_training import PilotExportRequest


def cuda_runtime_dependencies() -> dict[str, str]:
    """Read the committed metadata-only lock projection for independent CUDA use."""
    path = (
        Path(__file__).resolve().parents[1]
        / "infrastructure/cuda/runtime_dependencies.json"
    )
    pins = TypeAdapter(dict[str, str]).validate_python(
        json.loads(path.read_text(encoding="utf-8")), strict=True
    )
    if pins.get("torch") != "2.14.0+cu126" or not 1 <= len(pins) <= 128:
        raise ValueError("The committed CUDA dependency record is incompatible.")
    return pins


def export_pilot_checkpoint(request: PilotExportRequest) -> ModelExportSummary:
    """Export frozen CPU copies while keeping the selected runtime explicit."""
    from backend_service.pilot_operations import frozen_pilot_models

    root = Path(request.output_root).resolve()
    check_output_root(root)
    export_device(request.device)
    with pilot_execution(request) as deadline:
        encoder, decoder, state = frozen_pilot_models(
            Path(request.checkpoint), request.experiment_identifier, device="cpu"
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
            runtime_dependencies={"cuda": cuda_runtime_dependencies()},
            verification_device=request.device,
        )
        write_record(
            run_directory(root, request.experiment_identifier) / "exports.json", result
        )
        return result
