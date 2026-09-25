"""Shared request builders for the critic and fork tests."""

from pathlib import Path

import torch
from torch import nn

from schemas.pilot_training import (
    PilotCriticConfiguration,
    PilotTrainingConfiguration,
    PilotTrainingRequest,
)

ENABLED = PilotCriticConfiguration(enabled=True, weight=1.0)
ZERO_WEIGHT = PilotCriticConfiguration(enabled=True, weight=0.0)


def request(
    dataset: Path,
    root: Path,
    experiment: str = "critic_test",
    *,
    critic: PilotCriticConfiguration | None = None,
    steps: int = 10,
    resume_checkpoint: str | None = None,
    fork_from_checkpoint: str | None = None,
) -> PilotTrainingRequest:
    """Build a two-step smoke request with one CPU thread and optional critic."""
    return PilotTrainingRequest(
        experiment_identifier=experiment,
        output_root=str(root / "output"),
        dataset_directory=str(dataset),
        configuration=PilotTrainingConfiguration(
            planned_optimizer_steps=steps,
            cpu_threads=4,
            critic=critic or PilotCriticConfiguration(),
        ),
        stop_after_step=2,
        resume_checkpoint=resume_checkpoint,
        fork_from_checkpoint=fork_from_checkpoint,
    )


def same_weights(first: nn.Module, second: nn.Module) -> bool:
    """Compare every tensor of two modules exactly."""
    expected = first.state_dict()
    return all(
        torch.equal(value, expected[name])
        for name, value in second.state_dict().items()
    )
