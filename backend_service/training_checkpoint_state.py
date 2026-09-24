"""Safe Torch state serialization and deterministic CPU random-state recovery."""

import copy
import hashlib
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from backend_service.training_checkpoint_files import canonical_bytes


def random_state() -> dict[str, Any]:
    """Encode Python, NumPy, and Torch generators as safe tensors and primitives."""
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "kind": str(numpy_state[0]),
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gaussian": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
        if cast(Callable[[], bool], torch.cuda.is_initialized)()
        else [],
    }
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    """Restore global generators after models and optimizer have been restored."""
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state(
        (
            numpy_state["kind"],
            numpy_state["keys"].numpy().astype(np.uint32),
            numpy_state["position"],
            numpy_state["has_gaussian"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not cast(Callable[[], bool], torch.cuda.is_initialized)():
            raise ValueError(
                "CUDA random state needs the same initialized environment."
            )
        torch.cuda.set_rng_state_all(state["cuda"])


def frozen_checksum(configuration: dict[str, object]) -> str:
    """Bind a checkpoint to the exact resolved training configuration."""
    return hashlib.sha256(canonical_bytes(configuration)).hexdigest()


def safe_state(value: Any) -> None:
    """Reject executable objects and unsupported dictionary keys before saving."""
    if value is None or type(value) in (str, int, float, bool):
        return
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            safe_state(item)
        return
    if isinstance(value, dict):
        if not all(type(key) in (str, int) for key in value):
            raise ValueError("Checkpoint keys must be strings or integers.")
        for item in value.values():
            safe_state(item)
        return
    raise ValueError("Only safe tensor and primitive checkpoint state is supported.")


def load_state(path: Path) -> dict[str, Any]:
    """Load only the safe Torch subset; never enable arbitrary pickle execution."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "encoder",
        "decoder",
        "optimizer",
        "random_state",
        "global_step",
        "sampler_state",
        "configuration",
        "identities",
    }:
        raise ValueError("Checkpoint state fields are invalid.")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Checkpoint state version is unsupported.")
    if type(value["global_step"]) is not int or value["global_step"] < 0:
        raise ValueError("Checkpoint optimizer step is invalid.")
    if not all(
        isinstance(value[name], dict)
        for name in value
        if name not in {"schema_version", "global_step"}
    ):
        raise ValueError("Checkpoint state mappings are invalid.")
    safe_state(value)
    return cast(dict[str, Any], value)


def validate_restore(
    state: dict[str, Any],
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Probe model, optimizer, and generator compatibility before changing callers."""
    copies = copy.deepcopy((encoder, decoder, optimizer))
    for model, name in ((encoder, "encoder"), (decoder, "decoder")):
        expected = model.state_dict()
        if set(state[name]) != set(expected):
            raise ValueError("Checkpoint model fields are incompatible.")
        for key, reference in expected.items():
            actual = state[name][key]
            if (
                not isinstance(actual, torch.Tensor)
                or actual.shape != reference.shape
                or actual.dtype != reference.dtype
                or not bool(torch.isfinite(actual).all())
            ):
                raise ValueError("Checkpoint model tensors are incompatible.")
    copies[0].load_state_dict(state["encoder"], strict=True)
    copies[1].load_state_dict(state["decoder"], strict=True)
    copies[2].load_state_dict(state["optimizer"])
    for parameter, values in copies[2].state.items():
        for value in values.values():
            if isinstance(value, torch.Tensor) and (
                not bool(torch.isfinite(value).all())
                or (value.ndim != 0 and value.shape != parameter.shape)
            ):
                raise ValueError("Checkpoint optimizer tensors are incompatible.")
    previous = random_state()
    try:
        restore_random_state(state["random_state"])
    finally:
        restore_random_state(previous)
