"""Checkpoint retention and atomic failure protection for the previous recovery."""

import os
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.training_checkpoints import inspect_checkpoint, save_checkpoint
from schemas.checkpoints import CheckpointIndex, CheckpointMetrics, CheckpointSummary


@pytest.fixture(autouse=True)
def available_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make storage tests independent of the host's current disk utilization."""
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: usage._replace(total=200 * 1024**3, free=150 * 1024**3),
    )


def save(
    root: Path,
    step: int,
    recovery: float | None = None,
    *,
    pinned: bool = False,
) -> CheckpointSummary:
    """Save a minimal real optimizer state with optional measured ranking values."""
    encoder, decoder = nn.Linear(1, 1), nn.Linear(1, 1)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()])
    metrics = (
        None
        if recovery is None
        else CheckpointMetrics(
            exact_message_recovery=recovery,
            median_psnr=40.0,
            median_ssim=0.98,
        )
    )
    return save_checkpoint(
        root,
        encoder,
        decoder,
        optimizer,
        global_step=step,
        sampler_state={},
        configuration={"seed": 1},
        identities={"environment": "test"},
        metrics=metrics,
        pinned=pinned,
    )


def test_retains_latest_three_best_and_pinned(tmp_path: Path) -> None:
    """Latest recovery can be unmeasured while weaker pinned states survive."""
    pinned = save(tmp_path, 0, 0.01, pinned=True)
    candidates = [save(tmp_path, step, step / 10) for step in range(1, 6)]
    latest = save(tmp_path, 6)
    index = CheckpointIndex.model_validate_json((tmp_path / "index.json").read_bytes())
    wanted = {pinned.checkpoint_identifier, latest.checkpoint_identifier}
    wanted.update(item.checkpoint_identifier for item in candidates[-3:])
    assert {item.checkpoint_identifier for item in index.checkpoints} == wanted
    assert index.latest_identifier == latest.checkpoint_identifier
    assert {path.name for path in tmp_path.iterdir() if path.is_dir()} == wanted
    for item in index.checkpoints:
        assert inspect_checkpoint(tmp_path / item.checkpoint_identifier) == item


def test_flush_failure_preserves_previous_pointer_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure before publication leaves the complete earlier recovery untouched."""
    previous = save(tmp_path, 1)
    previous_index = (tmp_path / "index.json").read_bytes()

    def failed_flush(_descriptor: int) -> None:
        """Inject a filesystem failure with details that must never escape."""
        raise OSError("private path or disk failure details")

    monkeypatch.setattr(os, "fsync", failed_flush)
    with pytest.raises(ApplicationFailure) as failure:
        save(tmp_path, 2)
    assert "private" not in str(failure.value)
    assert (tmp_path / "index.json").read_bytes() == previous_index
    assert inspect_checkpoint(tmp_path / previous.checkpoint_identifier) == previous
    assert not list(tmp_path.glob("*.pending"))


def test_disk_reserve_failure_keeps_previous_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new checkpoint cannot spend protected headroom or damage old state."""
    previous = save(tmp_path, 1)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: usage._replace(total=200 * 1024**3, free=20 * 1024**3),
    )
    with pytest.raises(ApplicationFailure):
        save(tmp_path, 2)
    assert inspect_checkpoint(tmp_path / previous.checkpoint_identifier) == previous
