"""Explicit device selection for independent experimental model packages."""

import os
from collections.abc import Callable
from typing import Literal, cast

import torch

from backend_service.failures import ApplicationFailure


def configure_cuda_numerics() -> None:
    """Apply the same deterministic FP32 policy in training and isolated runtimes."""
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    initialized = cast(Callable[[], bool], torch.cuda.is_initialized)()
    if configured not in (None, ":4096:8") or (initialized and configured is None):
        raise ApplicationFailure(
            "cuda_numerical_environment",
            "Restart with CUBLAS_WORKSPACE_CONFIG=:4096:8 before using CUDA.",
            422,
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def export_device(device: Literal["cpu", "cuda"]) -> torch.device:
    """Reject unavailable CUDA without changing the requested execution device."""
    if device not in ("cpu", "cuda"):
        raise ApplicationFailure("model_device", "Select CPU or CUDA.", 422)
    if device == "cuda":
        configure_cuda_numerics()
        if not torch.cuda.is_available():
            raise ApplicationFailure(
                "cuda_unavailable",
                "CUDA is unavailable. Use the recorded CUDA environment and GPU host.",
                422,
            )
    return torch.device(device)
