"""Image-quality measurements on saved and reopened integer RGB pixels."""

import math
import resource
import sys

import torch
from torch import Tensor
from torch.nn import functional


def peak_process_memory_bytes() -> int:
    """Normalize the process lifetime peak memory counter across CPU platforms."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _window_mean(values: Tensor) -> Tensor:
    """Use an 11-pixel Gaussian window with sigma 1.5 and valid borders."""
    coordinates = torch.arange(-5, 6, dtype=values.dtype, device=values.device)
    window = torch.exp(-(coordinates.square()) / (2.0 * 1.5**2))
    window = window / window.sum()
    channels = values.shape[1]
    horizontal = window.view(1, 1, 1, 11).expand(channels, 1, 1, 11)
    vertical = window.view(1, 1, 11, 1).expand(channels, 1, 11, 1)
    first = functional.conv2d(values, horizontal, groups=channels)
    return functional.conv2d(first, vertical, groups=channels)


def image_quality(cover: Tensor, reopened: Tensor) -> tuple[float | None, float, bool]:
    """Return PSNR, mean RGB SSIM, and exact equality; perfect PSNR is null."""
    first, second = cover.double(), reopened.double()
    error = float((first - second).square().mean())
    identical = error == 0.0
    peak_ratio = None if identical else 10.0 * math.log10(1.0 / error)
    first_mean, second_mean = _window_mean(first), _window_mean(second)
    first_variance = (_window_mean(first.square()) - first_mean.square()).clamp_min(0)
    second_variance = (_window_mean(second.square()) - second_mean.square()).clamp_min(
        0
    )
    covariance = _window_mean(first * second) - first_mean * second_mean
    numerator = (2 * first_mean * second_mean + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (first_mean.square() + second_mean.square() + 0.01**2) * (
        first_variance + second_variance + 0.03**2
    )
    similarity = float((numerator / denominator).mean())
    return peak_ratio, min(1.0, max(-1.0, similarity)), identical
