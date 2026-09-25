"""Critic training state and losses that share one encoder forward per batch."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from backend_service.failures import ApplicationFailure
from backend_service.model_critic import DenseCritic, build_critic, clip_critic_weights
from schemas.pilot_training import PilotCriticConfiguration, PilotTrainingConfiguration


@dataclass
class CriticState:
    """Keep the critic, its optimizer, and its fixed settings together."""

    critic: DenseCritic
    optimizer: torch.optim.Adam
    configuration: PilotCriticConfiguration


def create_critic_state(
    configuration: PilotTrainingConfiguration, device: torch.device
) -> CriticState | None:
    """Build the critic only when the frozen profile turns it on."""
    settings = configuration.critic
    if not settings.enabled:
        return None
    critic = build_critic(configuration.seed, settings.hidden_channels)
    critic.to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=settings.learning_rate)
    return CriticState(critic, optimizer, settings)


def set_parameter_gradients(module: nn.Module, enabled: bool) -> None:
    """Switch whether the critic's own weights collect gradients."""
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def realism_loss(state: CriticState, quantized: Tensor) -> Tensor:
    """Push the encoder towards images the critic scores as real, critic frozen."""
    set_parameter_gradients(state.critic, False)
    try:
        scores: Tensor = state.critic(quantized)
        return -scores.mean()
    finally:
        set_parameter_gradients(state.critic, True)


def critic_loss(state: CriticState, cover: Tensor, quantized: Tensor) -> Tensor:
    """Score real covers above encoded images using the same quantized tensor."""
    fake: Tensor = state.critic(quantized.detach())
    real: Tensor = state.critic(cover.detach())
    return fake.mean() - real.mean()


def finish_critic_step(state: CriticState) -> None:
    """Apply the accumulated critic update and clip its weights."""
    state.optimizer.step()
    clip_critic_weights(state.critic, state.configuration.weight_clip)


def critic_state_dict(state: CriticState) -> dict[str, Any]:
    """Serialize the critic as auxiliary checkpoint state."""
    return {
        "critic": state.critic.state_dict(),
        "critic_optimizer": state.optimizer.state_dict(),
    }


def load_critic_state(state: CriticState, auxiliary: dict[str, Any]) -> None:
    """Restore a saved critic and optimizer, refusing shapes or values that differ."""
    try:
        expected = state.critic.state_dict()
        saved = auxiliary["critic"]
        if set(saved) != set(expected):
            raise ValueError("Critic fields differ.")
        for key, reference in expected.items():
            value = saved[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != reference.shape
                or value.dtype != reference.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("Critic tensors differ.")
        state.critic.load_state_dict(saved, strict=True)
        state.optimizer.load_state_dict(auxiliary["critic_optimizer"])
        # A fork may change the critic settings; the frozen profile always wins.
        for group in state.optimizer.param_groups:
            group["lr"] = state.configuration.learning_rate
        clip_critic_weights(state.critic, state.configuration.weight_clip)
    except (KeyError, ValueError, RuntimeError, TypeError):
        raise ApplicationFailure(
            "checkpoint_incompatible",
            "The saved critic state does not match the requested critic settings.",
        ) from None
