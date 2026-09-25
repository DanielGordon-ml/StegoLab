"""Optional adversarial critic that scores image realism; never part of an export."""

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

CRITIC_SEED_OFFSET = 0x5A17


def _critic_stage(input_channels: int, hidden_channels: int) -> nn.Sequential:
    """Build one padded convolution with the critic's activation and normalization."""
    return nn.Sequential(
        nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=False),
        nn.BatchNorm2d(hidden_channels, eps=1e-5, momentum=0.1),
    )


class DenseCritic(nn.Module):
    """Give one realism score per image from three hidden convolution stages."""

    def __init__(self, hidden_channels: int = 32) -> None:
        """Freeze the three-stage layout used by the SteganoGAN critic."""
        super().__init__()
        self.stages = nn.Sequential(
            _critic_stage(3, hidden_channels),
            _critic_stage(hidden_channels, hidden_channels),
            _critic_stage(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, image_rgb: Tensor) -> Tensor:
        """Average the score map so every image gets one number."""
        scores: Tensor = self.stages(image_rgb)
        return scores.mean(dim=(1, 2, 3))


def initialized_cuda_devices() -> list[int]:
    """List CUDA generators to restore; seeding touches them once CUDA is live."""
    if cast(Callable[[], bool], torch.cuda.is_initialized)():
        return list(range(torch.cuda.device_count()))
    return []


def build_critic(seed: int, hidden_channels: int = 32) -> DenseCritic:
    """Initialize a reproducible critic without touching the caller's random state."""
    with torch.random.fork_rng(devices=initialized_cuda_devices(), device_type="cuda"):
        torch.manual_seed((seed + CRITIC_SEED_OFFSET) % 2**63)
        return DenseCritic(hidden_channels).float().cpu()


def clip_critic_weights(critic: nn.Module, limit: float) -> None:
    """Keep every critic weight inside a fixed range after each update."""
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.clamp_(-limit, limit)
