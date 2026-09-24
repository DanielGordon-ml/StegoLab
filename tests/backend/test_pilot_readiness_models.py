"""Keep legacy proof checkpoints readable without training or quality scoring."""

from pathlib import Path

import pytest
import torch

from backend_service.failures import ApplicationFailure
from backend_service.model_networks import build_models
from backend_service.pilot_readiness_models import _learned_models
from backend_service.training_checkpoints import (
    read_checkpoint_state,
    save_checkpoint,
)
from schemas.pilot_readiness import GpuReadinessRequest
from schemas.training import TrainingConfiguration


def test_real_legacy_shaped_checkpoint_uses_proof_loader(tmp_path: Path) -> None:
    """A shared planned-step field cannot misroute the strict CPU-proof profile."""
    encoder, decoder = build_models(0)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()])
    configuration = TrainingConfiguration(cpu_threads=1)
    saved = save_checkpoint(
        tmp_path / "checkpoints",
        encoder,
        decoder,
        optimizer,
        global_step=0,
        sampler_state={},
        configuration=configuration.model_dump(mode="json"),
        identities={"experiment_identifier": "legacy_fixture"},
    )
    request = GpuReadinessRequest(
        dataset_directory="unused",
        experiment_identifier="readiness_fixture",
        learned_checkpoint=str(tmp_path / "checkpoints" / saved.checkpoint_identifier),
        learned_experiment_identifier="legacy_fixture",
    )
    restored = _learned_models(request)
    for original, actual in zip((encoder, decoder), restored, strict=True):
        assert next(actual.parameters()).device.type == "cpu"
        assert not actual.training
        for name, values in original.state_dict().items():
            assert torch.equal(values, actual.state_dict()[name])


def test_original_sprint_four_checkpoint_remains_readable() -> None:
    """Inspect existing learned weights when present without decoding any image."""
    directory = Path("checkpoints/sprint04_baseline")
    candidates = sorted(directory.glob("checkpoint_1000_*/metadata.json"))
    if not candidates:
        pytest.skip("The original Sprint 4 checkpoint is not installed.")
    checkpoint = candidates[0].parent
    state = read_checkpoint_state(checkpoint)
    assert state["configuration"]["architecture"] == "dense_cpu_v1"
    encoder, decoder = _learned_models(
        GpuReadinessRequest(
            dataset_directory="unused",
            experiment_identifier="legacy_read_check",
            learned_checkpoint=str(checkpoint),
            learned_experiment_identifier="sprint04_baseline",
        )
    )
    for role, model in (("encoder", encoder), ("decoder", decoder)):
        assert next(model.parameters()).device.type == "cpu"
        for name, values in model.state_dict().items():
            assert torch.equal(values, state[role][name])


def test_unknown_architecture_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never guess a checkpoint family from an overlapping configuration key."""
    monkeypatch.setattr(
        "backend_service.pilot_readiness_models.read_checkpoint_state",
        lambda path: {"configuration": {"architecture": "unknown"}},
    )
    with pytest.raises(ApplicationFailure) as caught:
        _learned_models(
            GpuReadinessRequest(
                dataset_directory="unused",
                experiment_identifier="read_check",
                learned_checkpoint="checkpoint",
                learned_experiment_identifier="original",
            )
        )
    assert caught.value.code == "pilot_checkpoint_architecture"
