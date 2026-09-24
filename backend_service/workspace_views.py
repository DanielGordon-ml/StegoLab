"""Private discovery adapters for public research workspace summaries."""

from pathlib import Path

from pydantic import TypeAdapter

from backend_service.failures import ApplicationFailure
from backend_service.workspace_eligibility import checkpoint_eligibility
from backend_service.workspace_registry import WorkspaceRegistry, safe_path
from backend_service.workspace_reports import (
    discover,
    evaluation_reports,
    read_metadata,
    report_metrics,
    report_time,
)
from schemas.checkpoints import CheckpointIndex
from schemas.model_exports import ModelExportManifest
from schemas.pilot_training import PilotTrainingRun
from schemas.training import TrainingConfiguration, TrainingRun
from schemas.workspace import (
    WorkspaceCheckpoint,
    WorkspaceExport,
    WorkspaceMetrics,
    WorkspaceRun,
)


def discover_runs(root: Path, registry: WorkspaceRegistry) -> list[WorkspaceRun]:
    """Join legacy separate evaluation folders using frozen model/data identities."""
    reports = evaluation_reports(root)
    result: list[WorkspaceRun] = []
    for path in discover(
        root, ("logs/*/training_run.json", ".runtime/*/logs/*/training_run.json")
    ):
        try:
            run: TrainingRun | PilotTrainingRun = TypeAdapter(
                TrainingRun | PilotTrainingRun
            ).validate_json(read_metadata(path))
            report = reports.get((run.compatibility_identifier, run.dataset_revision))
            result.append(
                WorkspaceRun(
                    identifier=registry.register("run", path, root),
                    experiment_identifier=run.experiment_identifier,
                    status=run.status,
                    global_step=run.global_step,
                    dataset_revision=run.dataset_revision,
                    created_at=report_time(path),
                    metrics=report_metrics(report) if report else WorkspaceMetrics(),
                )
            )
        except (OSError, ValueError, ApplicationFailure):
            continue
    return sorted(result, key=lambda item: item.created_at, reverse=True)


def discover_checkpoints(
    root: Path, registry: WorkspaceRegistry, datasets: dict[str, list[str]]
) -> list[WorkspaceCheckpoint]:
    """Verify indexed checkpoints before presenting exact-continuation blockers."""
    result: list[WorkspaceCheckpoint] = []
    for path in discover(
        root, ("checkpoints/*/index.json", ".runtime/*/checkpoints/*/index.json")
    ):
        try:
            index = CheckpointIndex.model_validate_json(read_metadata(path))
        except (OSError, ValueError, ApplicationFailure):
            continue
        for entry in index.checkpoints:
            try:
                directory = safe_path(path.parent / entry.checkpoint_identifier, root)
                summary, blockers = checkpoint_eligibility(directory, datasets, root)
                if summary != entry:
                    raise ValueError("Checkpoint index and metadata disagree.")
                metric = summary.metrics
                try:
                    cpu_threads = TrainingConfiguration.model_validate(
                        summary.configuration
                    ).cpu_threads
                except ValueError:
                    cpu_threads = None
                result.append(
                    WorkspaceCheckpoint(
                        identifier=registry.register("checkpoint", directory, root),
                        experiment_identifier=summary.identities.get(
                            "experiment_identifier", path.parent.name
                        ),
                        dataset_revision=summary.identities.get("dataset_revision", ""),
                        cpu_threads=cpu_threads,
                        global_step=summary.global_step,
                        created_at=summary.created_at,
                        latest=summary.checkpoint_identifier == index.latest_identifier,
                        pinned=summary.pinned,
                        resume_blockers=blockers,
                        metrics=WorkspaceMetrics(
                            exact_recovery_rate=metric.exact_message_recovery,
                            psnr=metric.median_psnr,
                            ssim=metric.median_ssim,
                        )
                        if metric
                        else WorkspaceMetrics(),
                    )
                )
            except (OSError, ValueError, ApplicationFailure):
                continue
    return sorted(result, key=lambda item: item.created_at, reverse=True)


def discover_exports(root: Path, registry: WorkspaceRegistry) -> list[WorkspaceExport]:
    """Register only completed independently manifested encoder/decoder packages."""
    result: list[WorkspaceExport] = []
    for path in discover(
        root, ("models/*/*/manifest.json", ".runtime/*/models/*/*/manifest.json")
    ):
        try:
            manifest = ModelExportManifest.model_validate_json(read_metadata(path))
            identifier = registry.register("export", path.parent, root)
            result.append(
                WorkspaceExport(
                    identifier=identifier,
                    name=path.parent.parent.name,
                    kind=manifest.role,
                    artifact_identifier=identifier,
                )
            )
        except (OSError, ValueError, ApplicationFailure):
            continue
    return result
