"""Complete checkpoint recovery, random continuation, and safe load boundaries."""

import random
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.training_checkpoints import (
    inspect_checkpoint,
    load_checkpoint,
    read_checkpoint_state,
    save_checkpoint,
)

IDENTITIES = {"environment": "cpu-test", "dataset": "frozen", "model": "tiny-v1"}
CONFIGURATION: dict[str, object] = {"seed": 21, "learning_rate": 0.001}


@pytest.fixture(autouse=True)
def available_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provide deterministic disk headroom without allocating large fixture files."""
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: usage._replace(total=200 * 1024**3, free=150 * 1024**3),
    )


def models() -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Exercise optimizer moments, BatchNorm buffers, and stochastic model layers."""
    encoder = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.2))
    decoder = nn.Linear(4, 1)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()], lr=0.001
    )
    return encoder, decoder, optimizer


def step(
    encoder: nn.Module, decoder: nn.Module, optimizer: torch.optim.Optimizer
) -> torch.Tensor:
    """Advance real training using all three saved random number generators."""
    optimizer.zero_grad()
    inputs = torch.randn(4, 4) + float(np.random.random()) + random.random()
    output = decoder(encoder(inputs))
    output.square().mean().backward()
    optimizer.step()
    return cast(torch.Tensor, output.detach().clone())


def test_exact_resume_preserves_optimizer_bn_sampler_and_random_state(
    tmp_path: Path,
) -> None:
    """Interrupted training produces the same next output and complete next state."""
    random.seed(21)
    np.random.seed(21)
    torch.manual_seed(21)
    encoder, decoder, optimizer = models()
    step(encoder, decoder, optimizer)
    sampler = {"position": 8, "epoch": 2, "generator": torch.get_rng_state()}
    saved = save_checkpoint(
        tmp_path,
        encoder,
        decoder,
        optimizer,
        global_step=1,
        sampler_state=sampler,
        configuration=CONFIGURATION,
        identities=IDENTITIES,
    )
    expected_output = step(encoder, decoder, optimizer)
    expected_encoder = {
        name: value.clone() for name, value in encoder.state_dict().items()
    }
    expected_decoder = {
        name: value.clone() for name, value in decoder.state_dict().items()
    }
    expected_optimizer = optimizer.state_dict()
    other_encoder, other_decoder, other_optimizer = models()
    restored = load_checkpoint(
        tmp_path / saved.checkpoint_identifier,
        other_encoder,
        other_decoder,
        other_optimizer,
        identities=IDENTITIES,
        configuration=CONFIGURATION,
    )
    assert restored.summary.global_step == 1
    assert restored.sampler_state["position"] == 8
    assert torch.equal(
        cast(torch.Tensor, restored.sampler_state["generator"]),
        cast(torch.Tensor, sampler["generator"]),
    )
    assert torch.equal(
        step(other_encoder, other_decoder, other_optimizer), expected_output
    )
    for name, value in other_encoder.state_dict().items():
        assert torch.equal(value, expected_encoder[name])
    for name, value in other_decoder.state_dict().items():
        assert torch.equal(value, expected_decoder[name])
    for key, values in other_optimizer.state_dict()["state"].items():
        for name, value in values.items():
            assert torch.equal(value, expected_optimizer["state"][key][name])


def test_read_for_export_is_verified_and_does_not_restore_random_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weights-only reads retain the caller's generator and never enable pickle."""
    encoder, decoder, optimizer = models()
    saved = save_checkpoint(
        tmp_path,
        encoder,
        decoder,
        optimizer,
        global_step=0,
        sampler_state={},
        configuration=CONFIGURATION,
        identities=IDENTITIES,
    )
    torch.manual_seed(101)
    before = torch.get_rng_state()
    original_load = torch.load
    flags: list[object] = []

    def checked_load(*arguments: Any, **keywords: Any) -> Any:
        """Observe the actual Torch loading policy while still deserializing bytes."""
        flags.append(keywords.get("weights_only"))
        return original_load(*arguments, **keywords)

    monkeypatch.setattr(torch, "load", checked_load)
    path = tmp_path / saved.checkpoint_identifier
    inspected = inspect_checkpoint(path)
    state = read_checkpoint_state(path)
    assert inspected.configuration == CONFIGURATION
    assert state["identities"] == IDENTITIES
    assert flags and all(flag is True for flag in flags)
    assert torch.equal(torch.get_rng_state(), before)


def test_incompatible_resume_changes_no_model_or_generator(tmp_path: Path) -> None:
    """Different data, configuration, or model shape fails before caller mutation."""
    encoder, decoder, optimizer = models()
    saved = save_checkpoint(
        tmp_path,
        encoder,
        decoder,
        optimizer,
        global_step=0,
        sampler_state={},
        configuration=CONFIGURATION,
        identities=IDENTITIES,
    )
    other_encoder, other_decoder, other_optimizer = models()
    incompatible_encoder = nn.Linear(5, 5)
    before = {name: value.clone() for name, value in other_encoder.state_dict().items()}
    random_before = torch.get_rng_state()
    for identities, configuration in (
        ({**IDENTITIES, "dataset": "changed"}, CONFIGURATION),
        (IDENTITIES, {**CONFIGURATION, "seed": 99}),
    ):
        with pytest.raises(ApplicationFailure):
            load_checkpoint(
                tmp_path / saved.checkpoint_identifier,
                other_encoder,
                other_decoder,
                other_optimizer,
                identities=identities,
                configuration=configuration,
            )
    with pytest.raises(ApplicationFailure):
        load_checkpoint(
            tmp_path / saved.checkpoint_identifier,
            incompatible_encoder,
            other_decoder,
            other_optimizer,
            identities=IDENTITIES,
        )
    for name, value in other_encoder.state_dict().items():
        assert torch.equal(value, before[name])
    assert torch.equal(torch.get_rng_state(), random_before)


def test_corrupt_checkpoint_and_metadata_are_rejected(tmp_path: Path) -> None:
    """No partial or altered bytes can be presented as a verified saved model."""
    encoder, decoder, optimizer = models()
    saved = save_checkpoint(
        tmp_path,
        encoder,
        decoder,
        optimizer,
        global_step=0,
        sampler_state={},
        configuration=CONFIGURATION,
        identities=IDENTITIES,
    )
    path = tmp_path / saved.checkpoint_identifier
    original = (path / "state.pt").read_bytes()
    (path / "state.pt").write_bytes(original[:-40])
    with pytest.raises(ApplicationFailure):
        inspect_checkpoint(path)
    (path / "state.pt").write_bytes(original)
    (path / "metadata.json").write_text("{broken")
    with pytest.raises(ApplicationFailure):
        inspect_checkpoint(path)
