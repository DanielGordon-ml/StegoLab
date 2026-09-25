"""Optional critic and full-state forking, built and tested without a GPU run."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch
from pilot_critic_fixtures import ENABLED, ZERO_WEIGHT, request, same_weights
from pilot_data_fixtures import pilot_revision

from backend_service.failures import ApplicationFailure
from backend_service.model_critic import build_critic, clip_critic_weights
from backend_service.model_networks import model_pair_identifier
from backend_service.pilot_operations import frozen_pilot_models
from backend_service.pilot_training import train_pilot
from backend_service.pilot_training_state import (
    create_pilot_session,
    pilot_training_step,
    save_pilot_session,
)
from backend_service.training_checkpoint_state import load_state
from backend_service.training_checkpoints import read_checkpoint_state, save_checkpoint


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Prepare one immutable synthetic revision shared by every test here."""
    return pilot_revision(tmp_path_factory.mktemp("critic") / "dataset")


def test_critic_shapes_gradients_and_untouched_random_state() -> None:
    """Score one number per image, train, clip, and leave the global generator alone."""
    torch.manual_seed(3)
    reference = torch.rand(4)
    torch.manual_seed(3)
    critic = build_critic(7)
    assert torch.equal(torch.rand(4), reference)
    if torch.cuda.is_available():
        cast(Callable[[], None], torch.cuda.init)()
        torch.manual_seed(3)
        cuda_reference = torch.cuda.get_rng_state()
        build_critic(7)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_reference)
    assert same_weights(critic, build_critic(7))
    assert not same_weights(critic, build_critic(8))
    scores = critic(torch.rand(2, 3, 64, 64, requires_grad=True))
    assert scores.shape == (2,)
    scores.sum().backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in critic.parameters()
    )
    with torch.no_grad():
        next(critic.parameters()).fill_(5.0)
    clip_critic_weights(critic, 0.1)
    limit = float(torch.tensor(0.1))
    assert max(float(p.abs().max()) for p in critic.parameters()) <= limit


def test_zero_weight_critic_leaves_the_pair_bit_identical(
    dataset: Path, tmp_path: Path
) -> None:
    """Enabling the critic with weight zero changes nothing in the encoder pair."""
    # Sessions share the process generator, so each one runs before the next starts.
    plain = create_pilot_session(request(dataset, tmp_path / "plain"))
    plain_losses = pilot_training_step(plain)
    after_one_step = {
        name: value.clone() for name, value in plain.encoder.state_dict().items()
    }
    plain_losses = pilot_training_step(plain)
    with_critic = create_pilot_session(
        request(dataset, tmp_path / "critic", critic=ZERO_WEIGHT)
    )
    for _ in range(2):
        critic_losses = pilot_training_step(with_critic)
    assert plain_losses[:3] == critic_losses[:3]
    assert plain_losses.critic_loss is None
    assert critic_losses.critic_loss is not None
    assert same_weights(plain.encoder, with_critic.encoder)
    assert same_weights(plain.decoder, with_critic.decoder)
    assert plain.sampler.state_dict() == with_critic.sampler.state_dict()
    assert model_pair_identifier(plain.encoder, plain.decoder) == (
        model_pair_identifier(with_critic.encoder, with_critic.decoder)
    )
    weighted = create_pilot_session(
        request(dataset, tmp_path / "weighted", critic=ENABLED)
    )
    pilot_training_step(weighted)
    assert not all(
        torch.equal(value, after_one_step[name])
        for name, value in weighted.encoder.state_dict().items()
    )


def test_critic_state_round_trips_through_a_version_two_checkpoint(
    dataset: Path, tmp_path: Path
) -> None:
    """Resume a critic run exactly, and keep the pair readable for evaluation."""
    original = request(dataset, tmp_path, critic=ENABLED)
    continuous = create_pilot_session(original)
    pilot_training_step(continuous)
    expected = pilot_training_step(continuous)
    interrupted = create_pilot_session(original)
    pilot_training_step(interrupted)
    saved = save_pilot_session(interrupted, tmp_path / "checkpoints")
    path = tmp_path / "checkpoints" / saved.checkpoint_identifier
    state = read_checkpoint_state(path)
    assert state["schema_version"] == 2
    assert set(state["auxiliary"]) == {"critic", "critic_optimizer"}
    resumed = create_pilot_session(
        original.model_copy(update={"resume_checkpoint": str(path)})
    )
    assert resumed.critic is not None and continuous.critic is not None
    assert pilot_training_step(resumed) == expected
    assert same_weights(resumed.critic.critic, continuous.critic.critic)
    assert same_weights(resumed.encoder, continuous.encoder)
    encoder, decoder, _ = frozen_pilot_models(path, "critic_test")
    assert model_pair_identifier(encoder, decoder) == model_pair_identifier(
        interrupted.encoder, interrupted.decoder
    )
    with pytest.raises(ApplicationFailure, match="incompatible"):
        create_pilot_session(request(dataset, tmp_path, resume_checkpoint=str(path)))
    plain = save_pilot_session(
        create_pilot_session(request(dataset, tmp_path / "plain")),
        tmp_path / "plain_checkpoints",
    )
    assert (
        read_checkpoint_state(
            tmp_path / "plain_checkpoints" / plain.checkpoint_identifier
        )["schema_version"]
        == 1
    )
    with pytest.raises(ApplicationFailure, match="incompatible"):
        create_pilot_session(
            request(
                dataset,
                tmp_path / "plain",
                critic=ENABLED,
                resume_checkpoint=str(
                    tmp_path / "plain_checkpoints" / plain.checkpoint_identifier
                ),
            )
        )
    # A checkpoint written before the critic existed carries no critic block.
    older = create_pilot_session(request(dataset, tmp_path / "older"))
    pre_critic = save_checkpoint(
        tmp_path / "older_checkpoints",
        older.encoder,
        older.decoder,
        older.optimizer,
        global_step=0,
        sampler_state=older.sampler.state_dict(),
        configuration={
            key: value
            for key, value in older.configuration.model_dump(mode="json").items()
            if key != "critic"
        },
        identities=older.identities,
    )
    older_path = str(tmp_path / "older_checkpoints" / pre_critic.checkpoint_identifier)
    assert (
        create_pilot_session(
            request(dataset, tmp_path / "older", resume_checkpoint=older_path)
        ).global_step
        == 0
    )
    with pytest.raises(ApplicationFailure, match="no critic state"):
        create_pilot_session(
            request(
                dataset,
                tmp_path / "older",
                critic=ENABLED,
                resume_checkpoint=older_path,
            )
        )


def test_cpu_smoke_with_critic_stays_inside_its_limit(
    dataset: Path, tmp_path: Path
) -> None:
    """Run the real smoke command with the critic on and record its loss."""
    smoke = request(dataset, tmp_path, "critic_smoke", critic=ENABLED)
    result = train_pilot(smoke)
    assert result.status == "completed" and result.global_step == 2
    root = Path(smoke.output_root)
    records = [
        json.loads(line)
        for path in root.rglob("steps.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(records) == 2 and all(
        record["critic_loss"] is not None for record in records
    )
    assert read_checkpoint_state(root / result.checkpoint)["schema_version"] == 2


def test_checkpoint_version_gate_rejects_unknown_or_mismatched_layouts(
    tmp_path: Path,
) -> None:
    """Accept only layout one without auxiliary state and layout two with it."""
    base: dict[str, object] = {
        "schema_version": 1,
        "encoder": {},
        "decoder": {},
        "optimizer": {},
        "random_state": {},
        "global_step": 0,
        "sampler_state": {},
        "configuration": {},
        "identities": {},
    }
    for changes in (
        {"schema_version": 3},
        {"schema_version": 2},
        {"schema_version": 1, "auxiliary": {}},
        {"schema_version": True},
    ):
        path = tmp_path / "state.pt"
        torch.save({**base, **changes}, path)
        with pytest.raises(ValueError):
            load_state(path)
    torch.save({**base, "schema_version": 2, "auxiliary": {}}, tmp_path / "two.pt")
    assert load_state(tmp_path / "two.pt")["schema_version"] == 2
