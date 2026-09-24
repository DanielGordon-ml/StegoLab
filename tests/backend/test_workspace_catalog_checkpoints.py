"""Exact checkpoint eligibility stays read-only and explains incompatible state."""

import shutil
from pathlib import Path

import pytest
import torch

from backend_service.model_networks import build_models
from backend_service.proof_runtime import (
    environment_identity,
    training_source_identity,
)
from backend_service.training_checkpoints import save_checkpoint
from backend_service.workspace_catalog import WorkspaceCatalog
from backend_service.workspace_eligibility import checkpoint_eligibility
from schemas.training import TrainingConfiguration


def test_checkpoint_checks_do_not_change_random_state_or_original_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current identities pass; changed code/data and finished runs block resume."""
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: usage._replace(total=200 * 1024**3, free=150 * 1024**3),
    )
    configuration = TrainingConfiguration(cpu_threads=1)
    encoder, decoder = build_models(0)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()])
    identities = {
        "dataset_revision": "a" * 64,
        "environment": environment_identity(configuration),
        "architecture": configuration.architecture,
        "experiment_identifier": "example",
        "training_source": training_source_identity(),
    }
    parent = tmp_path / "checkpoints" / "example"
    summary = save_checkpoint(
        parent,
        encoder,
        decoder,
        optimizer,
        global_step=1000,
        sampler_state={"next_cover": 0},
        configuration=configuration.model_dump(mode="json"),
        identities=identities,
    )
    path = parent / summary.checkpoint_identifier
    before = {file.name: file.read_bytes() for file in path.iterdir()}
    random_before = torch.get_rng_state()
    _, blockers = checkpoint_eligibility(path, {"a" * 64: []}, tmp_path)
    assert blockers == ["This experiment has completed all 1,000 planned steps."]
    assert torch.equal(random_before, torch.get_rng_state())
    assert before == {file.name: file.read_bytes() for file in path.iterdir()}
    catalog = WorkspaceCatalog(tmp_path, tmp_path / "private", source_roots={})
    published = catalog.snapshot().checkpoints[0]
    assert published.cpu_threads == 1 and published.dataset_revision == "a" * 64
    monkeypatch.setattr(
        "backend_service.workspace_eligibility.training_source_identity",
        lambda: "changed",
    )
    _, blockers = checkpoint_eligibility(path, {}, tmp_path)
    assert any("training code has changed" in item for item in blockers)
    assert any("dataset is unavailable" in item for item in blockers)
    save_checkpoint(
        parent,
        encoder,
        decoder,
        optimizer,
        global_step=1001,
        sampler_state={},
        configuration={"profile": "unsupported"},
        identities=identities,
    )
    assert catalog.snapshot().checkpoints[0].cpu_threads is None
