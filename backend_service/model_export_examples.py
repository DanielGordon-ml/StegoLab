"""Public deterministic tensor examples for independently checking exports."""

from io import BytesIO
from typing import Literal, cast

import numpy as np
import torch
from torch import nn

from backend_service.model_export_io import export_failure, tensor_array


def example_inputs(role: Literal["encoder", "decoder"]) -> tuple[torch.Tensor, ...]:
    """Build constant public RGB and alternating public payload bits."""
    image = torch.full((1, 3, 512, 512), 0.5, dtype=torch.float32)
    if role == "decoder":
        return (image,)
    payload = (torch.arange(512 * 512) % 2).float().reshape(1, 1, 512, 512)
    return image, payload


def example_output(model: nn.Module, role: Literal["encoder", "decoder"]) -> bytes:
    """Save a public eager reference with no plaintext or training data."""
    with torch.inference_mode():
        result = cast(torch.Tensor, model(*example_inputs(role)))
    if not torch.isfinite(result).all():
        raise export_failure()
    with BytesIO() as output:
        np.save(output, tensor_array(result), allow_pickle=False)
        return output.getvalue()


def verify_example(
    graph: nn.Module, role: Literal["encoder", "decoder"], expected: torch.Tensor
) -> None:
    """Require bounded floating-point agreement with the package reference."""
    with torch.inference_mode():
        actual = cast(torch.Tensor, graph(*example_inputs(role)))
    if actual.shape != expected.shape or not torch.allclose(
        expected, actual, rtol=1e-5, atol=1e-5
    ):
        raise export_failure()
