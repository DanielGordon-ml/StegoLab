"""Verified full-state CPU checkpoints with atomic publication and retention."""

import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, cast

import torch
from torch import nn

from backend_service.training_checkpoint_files import (
    ReservedWriter,
    atomic_json,
    checkpoint_failure,
    checkpoint_lock,
    checksum,
    read_json,
    sync_directory,
)
from backend_service.training_checkpoint_state import (
    frozen_checksum,
    load_state,
    random_state,
    restore_random_state,
    safe_state,
    validate_restore,
)
from schemas.checkpoints import CheckpointIndex, CheckpointMetrics, CheckpointSummary


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Return the restored step, frozen settings, and caller-owned sampler state."""

    summary: CheckpointSummary
    sampler_state: dict[str, object]
    configuration: dict[str, object]
    auxiliary: dict[str, Any]


def _index(directory: Path) -> CheckpointIndex:
    """Read the published retention index without inventing replacements."""
    path = directory / "index.json"
    if not path.exists() and not path.is_symlink():
        return CheckpointIndex()
    return CheckpointIndex.model_validate_json(read_json(path))


def _retained(index: CheckpointIndex, latest: CheckpointSummary) -> CheckpointIndex:
    """Keep latest recovery, three measured best candidates, and pinned states."""
    candidates = [*index.checkpoints, latest]
    measured = [item for item in candidates if item.metrics is not None]

    def rank(item: CheckpointSummary) -> tuple[float, float, float, int, str]:
        """Resolve metric ties deterministically in favor of a later step."""
        assert item.metrics is not None
        return (
            item.metrics.exact_message_recovery,
            item.metrics.median_psnr,
            item.metrics.median_ssim,
            item.global_step,
            item.checkpoint_identifier,
        )

    best = {item.checkpoint_identifier for item in sorted(measured, key=rank)[-3:]}
    best.add(latest.checkpoint_identifier)
    kept = [
        item for item in candidates if item.pinned or item.checkpoint_identifier in best
    ]
    return CheckpointIndex(
        latest_identifier=latest.checkpoint_identifier, checkpoints=kept
    )


def _read_checkpoint(path: Path) -> tuple[CheckpointSummary, dict[str, Any]]:
    """Cross-check safe serialized state with its independently stored manifest."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError("A completed checkpoint directory is required.")
    summary = CheckpointSummary.model_validate_json(read_json(path / "metadata.json"))
    if path.name != summary.checkpoint_identifier:
        raise ValueError("Checkpoint path and identity disagree.")
    state_path = path / summary.state_filename
    if checksum(state_path) != summary.state_checksum:
        raise ValueError("Checkpoint bytes changed.")
    if state_path.stat().st_size != summary.state_bytes:
        raise ValueError("Checkpoint size changed.")
    state = load_state(state_path)
    if (
        state["global_step"] != summary.global_step
        or state["identities"] != summary.identities
        or state["configuration"] != summary.configuration
        or frozen_checksum(state["configuration"]) != summary.configuration_checksum
    ):
        raise ValueError("Checkpoint state and metadata disagree.")
    return summary, state


def read_checkpoint_state(path: Path) -> dict[str, Any]:
    """Read verified weights for evaluation or export without changing generators."""
    try:
        return _read_checkpoint(path)[1]
    except Exception:
        raise checkpoint_failure() from None


def inspect_checkpoint(path: Path) -> CheckpointSummary:
    """Verify a completed checkpoint and return its safe public metadata."""
    try:
        return _read_checkpoint(path)[0]
    except Exception:
        raise checkpoint_failure() from None


def save_checkpoint(
    directory: Path,
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    sampler_state: dict[str, object],
    configuration: dict[str, object],
    identities: dict[str, str],
    metrics: CheckpointMetrics | None = None,
    pinned: bool = False,
    auxiliary: dict[str, Any] | None = None,
) -> CheckpointSummary:
    """Publish a verified full state before changing pointers or pruning old files.

    Auxiliary state, such as an optional critic, is written with the version-two
    layout; without it the version-one layout stays byte-compatible with older code.
    """
    stage: Path | None = None
    try:
        with checkpoint_lock(directory):
            previous = _index(directory)
            identifier = f"checkpoint_{global_step}_{uuid.uuid4().hex}"
            stage = directory / f".{identifier}.pending"
            stage.mkdir()
            state: dict[str, Any] = {
                "schema_version": 1,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "random_state": random_state(),
                "global_step": global_step,
                "sampler_state": sampler_state,
                "configuration": configuration,
                "identities": identities,
            }
            if auxiliary:
                state["schema_version"] = 2
                state["auxiliary"] = auxiliary
            safe_state(state)
            with (stage / "state.pt").open("xb") as stream:
                torch.save(state, cast(BinaryIO, ReservedWriter(stream, directory)))
                stream.flush()
                os.fsync(stream.fileno())
            checked = load_state(stage / "state.pt")
            validate_restore(checked, encoder, decoder, optimizer)
            summary = CheckpointSummary(
                checkpoint_identifier=identifier,
                global_step=global_step,
                created_at=datetime.now(UTC).isoformat(),
                configuration_checksum=frozen_checksum(configuration),
                state_checksum=checksum(stage / "state.pt"),
                state_bytes=(stage / "state.pt").stat().st_size,
                configuration=configuration,
                identities=identities,
                metrics=metrics,
                pinned=pinned,
            )
            atomic_json(stage / "metadata.json", summary.model_dump(mode="json"))
            sync_directory(stage)
            completed = directory / identifier
            os.rename(stage, completed)
            stage = None
            sync_directory(directory)
            _read_checkpoint(completed)
            retained = _retained(previous, summary)
            atomic_json(directory / "index.json", retained.model_dump(mode="json"))
            retained_names = {
                item.checkpoint_identifier for item in retained.checkpoints
            }
            for item in previous.checkpoints:
                if item.checkpoint_identifier not in retained_names:
                    old_path = directory / item.checkpoint_identifier
                    if old_path.is_symlink():
                        raise ValueError("Checkpoint directories cannot be links.")
                    shutil.rmtree(old_path)
            return summary
    except Exception:
        raise checkpoint_failure() from None
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def load_checkpoint(
    path: Path,
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    identities: dict[str, str],
    configuration: dict[str, object] | None = None,
) -> LoadedCheckpoint:
    """Restore complete training state only when frozen identities are compatible."""
    try:
        summary, state = _read_checkpoint(path)
        if summary.identities != identities:
            raise ValueError(
                "The checkpoint belongs to a different environment or data."
            )
        if configuration is not None and configuration != summary.configuration:
            raise ValueError("The checkpoint configuration is incompatible.")
        validate_restore(state, encoder, decoder, optimizer)
        encoder.load_state_dict(state["encoder"], strict=True)
        decoder.load_state_dict(state["decoder"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_random_state(state["random_state"])
        return LoadedCheckpoint(
            summary,
            state["sampler_state"],
            state["configuration"],
            state.get("auxiliary", {}),
        )
    except Exception:
        raise checkpoint_failure() from None
