"""Full-state forking of pilot checkpoints into new experiments."""

from pathlib import Path

import pytest
from pilot_critic_fixtures import ENABLED, request, same_weights
from pilot_data_fixtures import pilot_revision
from pydantic import ValidationError

from backend_service.failures import ApplicationFailure
from backend_service.pilot_training_state import (
    create_pilot_session,
    pilot_training_step,
    save_pilot_session,
)
from backend_service.training_checkpoints import save_checkpoint
from schemas.pilot_training import PilotCriticConfiguration, PilotTrainingRequest


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Prepare one immutable synthetic revision shared by every test here."""
    return pilot_revision(tmp_path_factory.mktemp("critic") / "dataset")


def test_fork_matches_resume_and_records_provenance(
    dataset: Path, tmp_path: Path
) -> None:
    """A fork without a critic continues exactly like a resume, under a new name."""
    original = request(dataset, tmp_path)
    continuous = create_pilot_session(original)
    pilot_training_step(continuous)
    expected = pilot_training_step(continuous)
    source = create_pilot_session(original)
    pilot_training_step(source)
    saved = save_pilot_session(source, tmp_path / "checkpoints")
    path = str(tmp_path / "checkpoints" / saved.checkpoint_identifier)
    forked = create_pilot_session(
        request(dataset, tmp_path, "critic_fork", fork_from_checkpoint=path)
    )
    assert forked.global_step == 1
    assert forked.identities["experiment_identifier"] == "critic_fork"
    assert forked.identities["forked_from"] == (
        f"critic_test:{saved.checkpoint_identifier}"
    )
    assert pilot_training_step(forked) == expected
    assert same_weights(forked.encoder, continuous.encoder)
    with_critic = create_pilot_session(
        request(
            dataset,
            tmp_path,
            "critic_fork_on",
            critic=ENABLED,
            fork_from_checkpoint=path,
        )
    )
    assert with_critic.critic is not None and with_critic.global_step == 1
    assert same_weights(with_critic.decoder, source.decoder)
    fork_saved = save_pilot_session(forked, tmp_path / "fork_checkpoints")
    resumed = create_pilot_session(
        request(
            dataset,
            tmp_path,
            "critic_fork",
            resume_checkpoint=str(
                tmp_path / "fork_checkpoints" / fork_saved.checkpoint_identifier
            ),
        )
    )
    assert resumed.identities["forked_from"] == forked.identities["forked_from"]
    assert resumed.global_step == 2


def test_fork_rejects_other_data_schedule_experiment_and_proof_checkpoints(
    dataset: Path, tmp_path: Path
) -> None:
    """Refuse forks that would silently change what the experiment measures."""
    original = request(dataset, tmp_path)
    session = create_pilot_session(original)
    pilot_training_step(session)
    saved = save_pilot_session(session, tmp_path / "checkpoints")
    path = str(tmp_path / "checkpoints" / saved.checkpoint_identifier)
    with pytest.raises(ApplicationFailure, match="new experiment identifier"):
        create_pilot_session(request(dataset, tmp_path, fork_from_checkpoint=path))
    with pytest.raises(ApplicationFailure, match="frozen settings"):
        create_pilot_session(
            request(dataset, tmp_path, "other", steps=20, fork_from_checkpoint=path)
        )
    other = pilot_revision(tmp_path / "other_dataset", tuning_count=3)
    with pytest.raises(ApplicationFailure, match="different dataset"):
        create_pilot_session(
            request(other, tmp_path, "other", fork_from_checkpoint=path)
        )
    proof = save_checkpoint(
        tmp_path / "proof",
        session.encoder,
        session.decoder,
        session.optimizer,
        global_step=1,
        sampler_state=session.sampler.state_dict(),
        configuration=session.configuration.model_dump(mode="json"),
        identities={**session.identities, "architecture": "dense_cpu_v1"},
    )
    with pytest.raises(ApplicationFailure, match="different dataset"):
        create_pilot_session(
            request(
                dataset,
                tmp_path,
                "other",
                fork_from_checkpoint=str(
                    tmp_path / "proof" / proof.checkpoint_identifier
                ),
            )
        )
    with pytest.raises(ValidationError, match="resume or fork"):
        PilotTrainingRequest.model_validate(
            original.model_dump()
            | {"resume_checkpoint": "a", "fork_from_checkpoint": "b"}
        )
    for invalid in ({"enabled": 1}, {"weight": 101}, {"hidden_channels": 16}):
        with pytest.raises(ValidationError):
            PilotCriticConfiguration.model_validate(invalid)


def test_fork_copies_a_critic_and_applies_the_requested_critic_settings(
    dataset: Path, tmp_path: Path
) -> None:
    """Forking a critic run continues it exactly, under the fork's own settings."""
    critic_request = request(dataset, tmp_path, critic=ENABLED)
    continuous = create_pilot_session(critic_request)
    pilot_training_step(continuous)
    expected = pilot_training_step(continuous)
    source = create_pilot_session(critic_request)
    pilot_training_step(source)
    saved = save_pilot_session(source, tmp_path / "checkpoints")
    path = str(tmp_path / "checkpoints" / saved.checkpoint_identifier)
    forked = create_pilot_session(
        request(
            dataset, tmp_path, "critic_copy", critic=ENABLED, fork_from_checkpoint=path
        )
    )
    assert forked.critic is not None and continuous.critic is not None
    assert source.critic is not None
    assert same_weights(forked.critic.critic, source.critic.critic)
    assert pilot_training_step(forked) == expected
    assert same_weights(forked.critic.critic, continuous.critic.critic)
    faster = PilotCriticConfiguration(enabled=True, weight=1.0, learning_rate=0.001)
    retuned = create_pilot_session(
        request(
            dataset,
            tmp_path,
            "critic_retuned",
            critic=faster,
            fork_from_checkpoint=path,
        )
    )
    assert retuned.critic is not None
    assert retuned.critic.optimizer.param_groups[0]["lr"] == 0.001
    assert same_weights(retuned.critic.critic, source.critic.critic)
