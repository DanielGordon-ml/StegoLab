"""Small dense convolution models and their fixed integer-image channel."""

import hashlib
import json
from typing import cast

import torch
from torch import Tensor, nn

MODEL_ARCHITECTURE = "dense_residual_v1"


def _hidden_stage(input_channels: int) -> nn.Sequential:
    """Build a padded convolution with the baseline activation and normalization."""
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=False),
        nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
    )


class DenseEncoder(nn.Module):
    """Add a learned RGB residual using one spatial binary payload channel."""

    def __init__(self) -> None:
        """Freeze four convolution stages and 32 hidden channels."""
        super().__init__()
        self.first = _hidden_stage(3)
        self.second = _hidden_stage(33)
        self.third = _hidden_stage(65)
        self.final = nn.Conv2d(97, 3, kernel_size=3, padding=1)

    def forward(self, cover_rgb: Tensor, payload_map: Tensor) -> Tensor:
        """Return raw residual-added RGB; the channel applies clamping separately."""
        first = self.first(cover_rgb)
        second = self.second(torch.cat((first, payload_map), dim=1))
        third = self.third(torch.cat((first, second, payload_map), dim=1))
        residual = self.final(torch.cat((first, second, third, payload_map), dim=1))
        return cast(Tensor, cover_rgb + residual)


class DenseDecoder(nn.Module):
    """Recover one logit per image pixel without a cover or encoder dependency."""

    def __init__(self) -> None:
        """Freeze the matching four-stage dense decoder."""
        super().__init__()
        self.first = _hidden_stage(3)
        self.second = _hidden_stage(32)
        self.third = _hidden_stage(64)
        self.final = nn.Conv2d(96, 1, kernel_size=3, padding=1)

    def forward(self, stego_rgb: Tensor) -> Tensor:
        """Return unthresholded logits for the full spatial payload channel."""
        first = self.first(stego_rgb)
        second = self.second(first)
        third = self.third(torch.cat((first, second), dim=1))
        return cast(Tensor, self.final(torch.cat((first, second, third), dim=1)))


def build_models(seed: int) -> tuple[DenseEncoder, DenseDecoder]:
    """Initialize a reproducible CPU pair without changing the caller's RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return DenseEncoder().float().cpu(), DenseDecoder().float().cpu()


def quantize_image(values: Tensor, *, straight_through: bool = False) -> Tensor:
    """Clamp then round ties to even on the 8-bit grid, optionally passing gradients."""
    clamped = values.clamp(0.0, 1.0)
    quantized = torch.round(clamped * 255.0) / 255.0
    if straight_through:
        return clamped + (quantized - clamped).detach()
    return quantized


def model_pair_identifier(encoder: nn.Module, decoder: nn.Module) -> str:
    """Bind tensor values, architecture, and normalization to one trusted pair."""
    digest = hashlib.sha256(b"StegoLab/dense_residual_v1/rgb_unit_interval\0")
    for name, model in (("encoder", encoder), ("decoder", decoder)):
        for key, value in sorted(model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            header = json.dumps(
                [name, key, str(tensor.dtype), list(tensor.shape)],
                separators=(",", ":"),
            ).encode("ascii")
            digest.update(len(header).to_bytes(4, "big"))
            digest.update(header)
            digest.update(tensor.numpy().tobytes())
    return "dense_v1_" + digest.hexdigest()
