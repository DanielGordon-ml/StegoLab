"""Export deterministic shared contracts or check for committed schema drift."""

import argparse
import json
from pathlib import Path

from backend_service.application import create_application
from schemas.capabilities import Capabilities, HealthStatus, ModelList
from schemas.checkpoints import CheckpointMetrics, CheckpointSummary
from schemas.configuration import (
    ConfigurationProfile,
    ConfigurationReset,
    ConfigurationUpdate,
)
from schemas.datasets import (
    DatasetImageRecord,
    DatasetManifest,
    DatasetPreparationRequest,
    DatasetRejection,
    DatasetSummary,
)
from schemas.errors import ApplicationError, ErrorEnvelope
from schemas.evaluation import EvaluationReport
from schemas.export_recovery import ExportRecoveryReport
from schemas.images import ImageSummary
from schemas.jobs import JobEvent, JobList, JobSnapshot
from schemas.model_exports import (
    ExportVerification,
    ModelExportManifest,
    ModelExportSummary,
)
from schemas.protocol import PayloadCapacity, ProtocolContext, ProtocolVerification
from schemas.training import (
    EvaluationRequest,
    ExportRequest,
    TrainingConfiguration,
    TrainingRequest,
    TrainingRun,
    TrainingStep,
)

ENTITY_MODELS = (
    TrainingConfiguration,
    TrainingRequest,
    EvaluationRequest,
    ExportRequest,
    TrainingRun,
    TrainingStep,
    CheckpointSummary,
    CheckpointMetrics,
    EvaluationReport,
    ExportRecoveryReport,
    ExportVerification,
    ModelExportManifest,
    ModelExportSummary,
    DatasetPreparationRequest,
    DatasetImageRecord,
    DatasetRejection,
    DatasetManifest,
    DatasetSummary,
    ApplicationError,
    ErrorEnvelope,
    Capabilities,
    HealthStatus,
    ModelList,
    ConfigurationProfile,
    ConfigurationUpdate,
    ConfigurationReset,
    JobSnapshot,
    JobList,
    JobEvent,
    ProtocolContext,
    PayloadCapacity,
    ProtocolVerification,
    ImageSummary,
)


def contract_documents() -> dict[Path, str]:
    """Build sorted JSON documents without initializing persistence."""
    documents = {Path("openapi.json"): create_application().openapi()}
    for model in ENTITY_MODELS:
        documents[Path("entities") / f"{model.__name__}.json"] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            **model.model_json_schema(mode="serialization"),
        }
    return {
        path: json.dumps(document, indent=2, sort_keys=True) + "\n"
        for path, document in documents.items()
    }


def export_contracts(directory: Path, check: bool = False) -> bool:
    """Write matching contracts, or return whether existing contracts match."""
    expected = contract_documents()
    if check:
        actual = set(directory.rglob("*.json")) if directory.exists() else set()
        if actual != {directory / path for path in expected}:
            return False
        return all(
            (directory / path).read_text(encoding="utf-8") == content
            for path, content in expected.items()
        )
    for path, content in expected.items():
        destination = directory / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    """Run the contract exporter from the project root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on contract drift.")
    parser.add_argument("--directory", type=Path, default=Path("contracts"))
    arguments = parser.parse_args()
    if not export_contracts(arguments.directory, arguments.check):
        parser.exit(
            1, "Shared contracts differ. Run the exporter and review the changes.\n"
        )


if __name__ == "__main__":
    main()
