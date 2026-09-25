"""Shared clock patching, constants and helpers for the ledger record tests."""

import shutil
from pathlib import Path

import pytest
from test_pilot_budget import Clock

from backend_service import pilot_budget, pilot_budget_records
from backend_service.pilot_budget import inspect_budget

LOST_START = 1_790_283_913.0
LOST_STOP = 1_790_287_542.0
EVIDENCE = (
    "CloudTrail RunInstances 2026-09-24T21:05:13Z and StopInstances "
    "2026-09-24T22:05:42Z for i-0ccaee67f0acaa574; no ledger session existed."
)


def patch_clocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Clock:
    """Patch both ledger modules' clocks and give tiny ledgers disk headroom."""
    value = Clock()
    monkeypatch.setattr(pilot_budget, "time", value)
    monkeypatch.setattr(pilot_budget_records, "time", value)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: usage._replace(total=200 * 1024**3, free=150 * 1024**3),
    )
    return value


def totals(directory: Path) -> dict[str, tuple[float, float, float]]:
    """Map each stage to its allocated, consumed and remaining seconds."""
    return {
        item.stage: (
            item.allocated_seconds,
            item.consumed_seconds,
            item.remaining_seconds,
        )
        for item in inspect_budget(directory).stage_totals
    }
