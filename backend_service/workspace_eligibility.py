"""Read-only eligibility checks for the fixed, bounded CPU research profile."""

from pathlib import Path

from backend_service.dataset_validation import validated_records
from backend_service.failures import ApplicationFailure
from backend_service.proof_runtime import (
    environment_identity,
    training_source_identity,
)
from backend_service.training_budget_records import BudgetLedger
from backend_service.training_checkpoint_files import read_json
from backend_service.training_checkpoint_state import frozen_checksum
from backend_service.training_checkpoints import inspect_checkpoint
from schemas.checkpoints import CheckpointSummary
from schemas.dataset_manifest import DatasetManifest
from schemas.training import TrainingConfiguration
from schemas.workspace import WorkspaceBudget


def dataset_eligibility(path: Path) -> tuple[DatasetManifest, list[str]]:
    """Check metadata and CPU-size limits without reading the full image corpus."""
    manifest, records = validated_records(path)
    blockers: list[str] = []
    if manifest.selected_images > 30:
        blockers.append("The CPU proof supports at most 30 source images.")
    if manifest.prepared_bytes > 512 * 1024**2:
        blockers.append("The CPU proof supports at most 512 MiB of prepared images.")
    for split in ("train", "tuning"):
        examples = [
            record
            for record in records
            if record.assigned_split == split
            and record.eligible
            and record.source_path == record.representative_source_path
        ][:4]
        if len(examples) != 4:
            blockers.append(f"At least four distinct {split} images are required.")
        if split == "tuning" and any(
            min(record.width, record.height) < 1024 for record in examples
        ):
            blockers.append("The first four tuning images need 1,024-pixel sides.")
    return manifest, blockers


def checkpoint_eligibility(
    path: Path, datasets: dict[str, list[str]], root: Path
) -> tuple[CheckpointSummary, list[str]]:
    """Verify saved bytes and exact resume identities without restoring generators."""
    summary = inspect_checkpoint(path)
    blockers: list[str] = []
    try:
        configuration = TrainingConfiguration.model_validate(summary.configuration)
    except ValueError:
        return summary, ["This checkpoint does not use the supported CPU profile."]
    identities = summary.identities
    if identities.get("architecture") != configuration.architecture:
        blockers.append("The saved architecture does not match the CPU profile.")
    if frozen_checksum(summary.configuration) != summary.configuration_checksum:
        blockers.append("The saved configuration checksum does not match.")
    if identities.get("environment") != environment_identity(configuration):
        blockers.append("Exact resume requires the original software environment.")
    if identities.get("training_source") != training_source_identity():
        blockers.append("The training code has changed since this checkpoint.")
    revision = identities.get("dataset_revision", "")
    if revision not in datasets:
        blockers.append("The original prepared dataset is unavailable.")
    else:
        blockers.extend(datasets[revision])
    experiment = identities.get("experiment_identifier", "")
    if path.parent != root / "checkpoints" / experiment:
        blockers.append("Resume requires this experiment's original output folder.")
    if summary.global_step >= configuration.planned_optimizer_steps:
        blockers.append("This experiment has completed all 1,000 planned steps.")
    expected = {
        "architecture",
        "dataset_revision",
        "environment",
        "training_source",
        "experiment_identifier",
    }
    if set(identities) != expected:
        blockers.append("The saved identity fields do not match this CPU profile.")
    return summary, blockers


def read_budget(root: Path) -> WorkspaceBudget:
    """Report original accounting conservatively without reconciling or resetting it."""
    directory = root / "state" / "cpu_proof"
    try:
        ledger_path = directory / "ledger.json"
        if ledger_path.exists() or ledger_path.is_symlink():
            ledger = BudgetLedger.model_validate_json(read_json(ledger_path))
        elif (directory / ".initialized.json").exists() or any(
            (root / "checkpoints").glob("*/index.json")
        ):
            raise ValueError
        else:
            ledger = BudgetLedger()
        reserved = ledger.active.reserved_seconds if ledger.active else 0.0
        remaining = max(
            0.0, ledger.maximum_total_seconds - ledger.consumed_seconds - reserved
        )
        reason = None
        if ledger.active is not None:
            reason = "A CPU operation is active or needs accounting recovery."
        elif remaining <= ledger.save_reserve_seconds:
            reason = "The CPU training allowance is exhausted; save time is reserved."
        return WorkspaceBudget(
            remaining_seconds=remaining,
            remaining_experiments=ledger.maximum_experiments - len(ledger.experiments),
            blocked_reason=reason,
        )
    except (OSError, ValueError, ApplicationFailure):
        return WorkspaceBudget(
            remaining_seconds=0.0,
            remaining_experiments=0,
            blocked_reason="The existing CPU budget could not be verified. "
            "Restore its original ledger before starting work.",
        )
