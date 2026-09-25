"""Copy the full training state of one experiment into a new experiment."""

from pathlib import Path
from typing import TYPE_CHECKING

from backend_service.failures import ApplicationFailure
from backend_service.pilot_training_critic import load_critic_state
from backend_service.training_checkpoints import inspect_checkpoint, load_checkpoint

if TYPE_CHECKING:
    from backend_service.pilot_training_state import PilotTrainingSession

FORK_EXCLUDED_IDENTITIES = ("experiment_identifier", "forked_from")


def fork_failure(message: str) -> ApplicationFailure:
    """Explain why a checkpoint cannot seed a new experiment."""
    return ApplicationFailure("fork_incompatible", message, 422)


def comparable_configuration(configuration: dict[str, object]) -> dict[str, object]:
    """Compare frozen settings without the critic block, which a fork may change."""
    return {key: value for key, value in configuration.items() if key != "critic"}


def fork_pilot_session(session: "PilotTrainingSession", source: Path) -> None:
    """Load everything a resume loads, then continue under a new experiment name."""
    summary = inspect_checkpoint(source)
    saved = {
        key: value
        for key, value in summary.identities.items()
        if key not in FORK_EXCLUDED_IDENTITIES
    }
    expected = {
        key: value
        for key, value in session.identities.items()
        if key not in FORK_EXCLUDED_IDENTITIES
    }
    if saved != expected:
        raise fork_failure(
            "The checkpoint was trained with a different dataset, selection, "
            "architecture, environment, or code. Fork only a pilot checkpoint "
            "that matches this profile."
        )
    if comparable_configuration(summary.configuration) != comparable_configuration(
        session.configuration.model_dump(mode="json")
    ):
        raise fork_failure(
            "The checkpoint's frozen settings differ from this request. Only the "
            "critic block may change when forking."
        )
    if (
        summary.identities.get("experiment_identifier")
        == (session.identities["experiment_identifier"])
    ):
        raise fork_failure(
            "Forking needs a new experiment identifier. Use resume to continue "
            "the same experiment."
        )
    loaded = load_checkpoint(
        source,
        session.encoder,
        session.decoder,
        session.optimizer,
        identities=summary.identities,
    )
    session.sampler.load_state_dict(loaded.sampler_state)
    session.global_step = loaded.summary.global_step
    session.identities["forked_from"] = (
        f"{summary.identities.get('experiment_identifier', '')}:"
        f"{summary.checkpoint_identifier}"
    )
    if session.critic is not None and "critic" in loaded.auxiliary:
        load_critic_state(session.critic, loaded.auxiliary)
