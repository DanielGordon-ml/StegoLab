"""Workspace discovery preserves original data and uses frozen report identities."""

from pathlib import Path

import pytest
from workspace_fixtures import evaluation_report, prepared_revision

from backend_service import workspace_eligibility
from backend_service.dataset_validation import validated_records
from backend_service.failures import ApplicationFailure
from backend_service.workspace_catalog import WorkspaceCatalog
from backend_service.workspace_eligibility import dataset_eligibility
from schemas.pilot_training import PilotTrainingRun
from schemas.training import TrainingRun


def test_prepared_dataset_is_bounded_and_registered_persistently(
    tmp_path: Path,
) -> None:
    """Inspect real metadata without claiming full pixels or changing its bytes."""
    root = tmp_path / "workspace"
    directory = prepared_revision(root)
    before = (directory / "manifest.json").read_bytes()
    catalog = WorkspaceCatalog(root, tmp_path / "state", source_roots={})
    snapshot = catalog.snapshot()
    assert len(snapshot.datasets) == 1
    dataset = snapshot.datasets[0]
    assert dataset.compatible and dataset.integrity == "not_checked"
    assert (dataset.training_images, dataset.tuning_images) == (4, 4)
    assert catalog.resolve_dataset(dataset.identifier) == directory
    assert str(tmp_path) not in snapshot.model_dump_json()
    reopened = WorkspaceCatalog(root, tmp_path / "state", source_roots={})
    assert reopened.snapshot().datasets[0].identifier == dataset.identifier
    assert (directory / "manifest.json").read_bytes() == before
    assert not (root / "state").exists()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("selected_images", 31, "30 source images"),
        ("prepared_bytes", 512 * 1024**2 + 1, "512 MiB"),
    ],
)
def test_cpu_profile_limits_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
    message: str,
) -> None:
    """Large-corpus metadata is visible but cannot start the bounded CPU profile."""
    directory = prepared_revision(tmp_path)
    manifest, records = validated_records(directory)
    monkeypatch.setattr(
        workspace_eligibility,
        "validated_records",
        lambda _: (manifest.model_copy(update={field: value}), records),
    )
    assert any(message in item for item in dataset_eligibility(directory)[1])


def test_legacy_evaluation_matches_model_and_revision_not_folder(
    tmp_path: Path,
) -> None:
    """Separate legacy evaluation directories attach only to their exact model run."""
    compatibility = "dense_v1_" + "a" * 64
    revision = "b" * 64
    run = TrainingRun(
        experiment_identifier="baseline",
        status="completed",
        global_step=1000,
        checkpoint="checkpoints/baseline/checkpoint",
        dataset_revision=revision,
        compatibility_identifier=compatibility,
        elapsed_seconds=1.0,
        remaining_experiment_seconds=100.0,
        selected_training_identities=[],
        selected_tuning_identities=[],
    )
    training = tmp_path / "logs" / "2026-09-23_21-41-42_baseline_first"
    separate = tmp_path / "logs" / "2026-09-23_22-00-00_other_name_second"
    training.mkdir(parents=True)
    separate.mkdir()
    (training / "training_run.json").write_text(run.model_dump_json())
    (separate / "evaluation.json").write_text(
        evaluation_report(compatibility, revision).model_dump_json()
    )
    catalog = WorkspaceCatalog(tmp_path, tmp_path / "private", source_roots={})
    observed = catalog.snapshot().runs[0]
    assert observed.metrics.trial_count == 1 and observed.metrics.psnr == 26.22
    (separate / "evaluation.json").write_text(
        evaluation_report("another_model", revision).model_dump_json()
    )
    assert catalog.snapshot().runs[0].metrics.trial_count is None


def test_sources_reject_escapes_links_and_stale_references(tmp_path: Path) -> None:
    """An opaque identifier never grants arbitrary filesystem access."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "images").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (source / "link").symlink_to(outside, target_is_directory=True)
    catalog = WorkspaceCatalog(
        tmp_path / "workspace",
        tmp_path / "state",
        source_roots={"Public examples": source},
    )
    identifier = catalog.snapshot().sources[0].identifier
    assert catalog.resolve_source(identifier, "images") == source / "images"
    for invalid in ("../outside", str(outside), "link", "images/../../outside"):
        with pytest.raises(ApplicationFailure):
            catalog.resolve_source(identifier, invalid)
    with pytest.raises(ApplicationFailure):
        catalog.resolve_checkpoint(identifier)
    moved = tmp_path / "moved"
    source.rename(moved)
    source.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ApplicationFailure):
        catalog.resolve_source(identifier)


def test_existing_cpu_smoke_remains_visible_without_quality_claims(
    tmp_path: Path,
) -> None:
    """Version-two diagnostic work stays discoverable with unmeasured metrics."""
    run = PilotTrainingRun(
        experiment_identifier="cpu_smoke",
        execution_mode="cpu_smoke",
        status="completed",
        global_step=2,
        checkpoint="checkpoints/cpu_smoke/state",
        dataset_revision="a" * 64,
        compatibility_identifier="dense_v1_" + "b" * 64,
        elapsed_seconds=1.0,
    )
    directory = tmp_path / ".runtime" / "pilot_smoke" / "logs" / "example"
    directory.mkdir(parents=True)
    (directory / "training_run.json").write_text(run.model_dump_json())
    catalog = WorkspaceCatalog(tmp_path, tmp_path / "private", source_roots={})
    observed = catalog.snapshot().runs[0]
    assert observed.experiment_identifier == "cpu_smoke"
    assert observed.global_step == 2 and observed.metrics.trial_count is None
