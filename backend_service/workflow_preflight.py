"""Read-only eligibility checks and private CLI request resolution."""

from pathlib import Path
from typing import TYPE_CHECKING

from backend_service.failures import ApplicationFailure
from backend_service.training_budget_records import BudgetLedger
from backend_service.training_checkpoint_files import read_json
from backend_service.workflow_outputs import output_blockers
from schemas.checkpoints import CheckpointSummary
from schemas.training import TrainingConfiguration
from schemas.workflows import TrainingPreflight, WorkflowRequest

if TYPE_CHECKING:
    from backend_service.workspace_catalog import WorkspaceCatalog


def checkpoint_summary(path: Path) -> CheckpointSummary:
    """Read bounded published metadata without loading model weights."""
    return CheckpointSummary.model_validate_json(read_json(path / "metadata.json"))


def read_budget(root: Path) -> BudgetLedger:
    """Inspect the original ledger without creating or reconciling an allowance."""
    directory = root / "state" / "cpu_proof"
    if (directory / "ledger.json").exists() or (directory / "ledger.json").is_symlink():
        return BudgetLedger.model_validate_json(read_json(directory / "ledger.json"))
    if (directory / ".initialized.json").exists() or any(
        (root / "checkpoints").glob("*/index.json")
    ):
        raise ValueError("The saved proof ledger is missing.")
    return BudgetLedger()


def training_preflight(
    catalog: "WorkspaceCatalog", request: WorkflowRequest
) -> TrainingPreflight:
    """Explain blockers while retaining the original fixed CPU profile and limits."""
    blockers = output_blockers(
        catalog.root, request.operation, request.experiment_identifier
    )
    warnings = [
        "CPU experiments do not establish release quality or detection resistance."
    ]
    configuration = TrainingConfiguration(cpu_threads=request.cpu_threads)
    remaining = 0.0
    slots = 0
    try:
        if request.operation != "train":
            blockers.append("Choose a training request for this check.")
        if request.dataset_identifier is None:
            blockers.append("Select a prepared dataset.")
        else:
            blockers.extend(catalog.dataset_blockers(request.dataset_identifier))
        summary = None
        if request.checkpoint_identifier is not None:
            blockers.extend(catalog.checkpoint_blockers(request.checkpoint_identifier))
            summary = checkpoint_summary(
                catalog.resolve_checkpoint(request.checkpoint_identifier)
            )
            if (
                summary.identities.get("experiment_identifier")
                != request.experiment_identifier
            ):
                blockers.append(
                    "Resume must use the checkpoint's original experiment name."
                )
            if summary.configuration != configuration.model_dump(mode="json"):
                blockers.append(
                    "Resume must keep every saved training setting unchanged."
                )
            if request.stop_after_step <= summary.global_step:
                blockers.append("Choose a stop step later than the saved checkpoint.")
            if request.dataset_identifier is not None:
                from backend_service.dataset_validation import load_manifest

                manifest = load_manifest(
                    catalog.resolve_dataset(request.dataset_identifier)
                )
                if manifest.revision != summary.identities.get("dataset_revision"):
                    blockers.append(
                        "Resume must use the checkpoint's original dataset revision."
                    )
        ledger = read_budget(catalog.root)
        slots = ledger.maximum_experiments - len(ledger.experiments)
        experiment = next(
            (
                item
                for item in ledger.experiments
                if item.experiment_identifier == request.experiment_identifier
            ),
            None,
        )
        remaining = max(
            0.0,
            min(
                ledger.maximum_total_seconds - ledger.consumed_seconds,
                ledger.maximum_experiment_seconds
                - (experiment.consumed_seconds if experiment else 0),
            ),
        )
        if experiment is None and (slots == 0 or summary is not None):
            blockers.append(
                "No matching experiment allowance is available in the original ledger."
            )
        if experiment is not None and summary is None:
            directory = catalog.root / "checkpoints" / request.experiment_identifier
            if directory.exists() and any(
                path.name != ".checkpoint.lock" for path in directory.iterdir()
            ):
                blockers.append(
                    "This experiment has saved state. Select its checkpoint to resume."
                )
        if remaining <= ledger.save_reserve_seconds:
            blockers.append(
                "The CPU training allowance is exhausted; "
                "remaining time is reserved for checks."
            )
        if ledger.active is not None:
            warnings.append(
                "A CPU allowance is active or needs recovery. "
                "The original budget will be checked again before work starts."
            )
    except ApplicationFailure as failure:
        blockers.append(failure.message)
    except (OSError, ValueError):
        blockers.append(
            "Saved data or the CPU allowance could not be verified. "
            "Keep the original files."
        )
    return TrainingPreflight(
        allowed=not blockers,
        blockers=list(dict.fromkeys(blockers)),
        warnings=warnings,
        resolved_settings=configuration.model_dump(mode="json"),
        remaining_budget_seconds=remaining,
        remaining_experiment_slots=slots,
    )


def resolve_request(
    catalog: "WorkspaceCatalog", request: WorkflowRequest
) -> tuple[str, dict[str, object]]:
    """Resolve references privately, retaining the original proof ledger."""
    blockers = output_blockers(
        catalog.root, request.operation, request.experiment_identifier
    )
    if blockers:
        raise ApplicationFailure("output_unavailable", " ".join(blockers), 422)
    document: dict[str, object] = {
        "schema_version": 1,
        "experiment_identifier": request.experiment_identifier,
        "output_root": str(catalog.root),
    }
    if request.operation == "prepare_dataset":
        if request.source_identifier is None:
            raise ApplicationFailure(
                "source_required", "Select a mounted local source.", 422
            )
        source = catalog.resolve_source(
            request.source_identifier, request.source_subdirectory
        )
        return "prepare_dataset", {
            "schema_version": 1,
            "source_kind": "local",
            "metadata_file": None,
            "source_directory": str(source),
            "output_root": str(catalog.root / "datasets"),
            "dataset_name": request.dataset_name,
            "seed": 0,
        }
    ledger = read_budget(catalog.root)
    if request.operation != "train" and not any(
        item.experiment_identifier == request.experiment_identifier
        for item in ledger.experiments
    ):
        raise ApplicationFailure(
            "experiment_unavailable",
            "The original experiment allowance is missing.",
            422,
        )
    if request.operation in ("train", "evaluate"):
        if request.dataset_identifier is None:
            raise ApplicationFailure(
                "dataset_required", "Select a prepared dataset.", 422
            )
        document["dataset_directory"] = str(
            catalog.resolve_dataset(request.dataset_identifier)
        )
    if request.checkpoint_identifier is not None:
        checkpoint = catalog.resolve_checkpoint(request.checkpoint_identifier)
        summary = checkpoint_summary(checkpoint)
        try:
            TrainingConfiguration.model_validate(summary.configuration)
        except ValueError:
            raise ApplicationFailure(
                "checkpoint_profile",
                "Select a checkpoint from the fixed CPU profile.",
                422,
            ) from None
        if (
            summary.identities.get("experiment_identifier")
            != request.experiment_identifier
        ):
            raise ApplicationFailure(
                "experiment_mismatch",
                "Use the checkpoint's original experiment name.",
                422,
            )
        document[
            "resume_checkpoint" if request.operation == "train" else "checkpoint"
        ] = str(checkpoint)
        if request.operation == "evaluate" and request.dataset_identifier is not None:
            from backend_service.dataset_validation import load_manifest

            blockers = catalog.dataset_blockers(request.dataset_identifier)
            manifest = load_manifest(
                catalog.resolve_dataset(request.dataset_identifier)
            )
            if manifest.revision != summary.identities.get("dataset_revision"):
                blockers.append("Use the dataset revision recorded in the checkpoint.")
            if blockers:
                raise ApplicationFailure(
                    "dataset_incompatible", " ".join(blockers), 422
                )
    elif request.operation != "train":
        raise ApplicationFailure(
            "checkpoint_required", "Select a saved checkpoint.", 422
        )
    if request.operation == "train":
        document["configuration"] = TrainingConfiguration(
            cpu_threads=request.cpu_threads
        ).model_dump(mode="json")
        document["stop_after_step"] = request.stop_after_step
    elif request.operation == "evaluate":
        document["lightweight"] = request.lightweight
    else:
        document["export_name"] = request.export_name
    return (
        "export_models" if request.operation == "export" else request.operation,
        document,
    )
