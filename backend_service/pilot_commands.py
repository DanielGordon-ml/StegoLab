"""Version-two pilot command routing without enabling public model operations."""

from schemas.base import StrictRecord
from schemas.pilot_preflight import PilotPreflightRequest
from schemas.pilot_training import (
    PilotEvaluationRequest,
    PilotExportRequest,
    PilotTrainingRequest,
)


def dispatch_pilot(command: str, document: str) -> StrictRecord:
    """Validate a complete request before importing its execution service."""
    if command == "preflight_pilot":
        from backend_service.pilot_preflight import preflight_pilot

        return preflight_pilot(PilotPreflightRequest.model_validate_json(document))
    if command == "train":
        from backend_service.pilot_training import train_pilot

        return train_pilot(PilotTrainingRequest.model_validate_json(document))
    if command == "evaluate":
        from backend_service.pilot_operations import evaluate_pilot_checkpoint

        return evaluate_pilot_checkpoint(
            PilotEvaluationRequest.model_validate_json(document)
        )
    from backend_service.pilot_export import export_pilot_checkpoint

    return export_pilot_checkpoint(PilotExportRequest.model_validate_json(document))
