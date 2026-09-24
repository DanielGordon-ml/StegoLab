"""Exercise deadline and process actions with fake clocks and subprocesses only."""

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend_service import pilot_budget, pilot_guard
from backend_service.pilot_budget import inspect_budget, start_session
from backend_service.pilot_budget_storage import initialize_pilot_budget, ledger_lock
from scripts.pilot_budget_guard import HostStopActions


@dataclass
class Actions:
    """Record guard requests without touching a real process or host."""

    clock: SimpleNamespace
    fail_save: bool = False
    events: list[tuple[str, float]] = field(default_factory=list)

    def request_checkpoint(self) -> None:
        """Record safe-save time and optionally model a dead application."""
        self.events.append(("save", self.clock.monotonic()))
        if self.fail_save:
            raise OSError("simulated missing process")

    def poweroff(self) -> None:
        """Record a poweroff decision without invoking the operating system."""
        self.events.append(("poweroff", self.clock.monotonic()))


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Advance simulated seconds on sleep so short-budget drills finish instantly."""
    value = SimpleNamespace(seconds=0.0, wall_offset=0.0)
    value.time = lambda: 1_000_000.0 + value.seconds + value.wall_offset
    value.monotonic = lambda: value.seconds

    def sleep(seconds: float) -> None:
        """Advance fake time rather than sleeping or spending GPU budget."""
        value.seconds += seconds

    value.sleep = sleep
    monkeypatch.setattr(pilot_budget, "time", value)
    monkeypatch.setattr(pilot_guard, "time", value)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: usage._replace(total=200 * 1024**3, free=150 * 1024**3),
    )
    return value


@pytest.mark.parametrize("fail_save", [False, True])
def test_short_budget_signals_then_powers_off_even_if_save_fails(
    tmp_path: Path, clock: SimpleNamespace, fail_save: bool
) -> None:
    """A stuck/missing application cannot postpone the independent stop deadline."""
    initialize_pilot_budget(tmp_path)
    start_session(
        tmp_path, "drill", "i-12345678", "setup", clock.time(), requested_seconds=600
    )
    actions = Actions(clock, fail_save)
    pilot_guard.run_guard(tmp_path, "drill", actions)
    assert actions.events == [("save", 300), ("poweroff", 480)]
    summary = inspect_budget(tmp_path)
    assert summary.active_session_identifier == "drill"
    assert summary.consumed_seconds == 480


def test_clock_regression_fails_closed_with_immediate_stop(
    tmp_path: Path, clock: SimpleNamespace
) -> None:
    """Restart/accounting uncertainty does not turn into additional paid time."""
    initialize_pilot_budget(tmp_path)
    start_session(
        tmp_path, "drill", "i-12345678", "setup", clock.time(), requested_seconds=600
    )

    def backwards_sleep(seconds: float) -> None:
        """Move wall time backwards while elapsed time still advances."""
        clock.seconds += seconds
        clock.wall_offset = -100

    clock.sleep = backwards_sleep
    actions = Actions(clock)
    pilot_guard.run_guard(tmp_path, "drill", actions)
    assert actions.events == [("save", 5), ("poweroff", 5)]


def test_ledger_lock_contention_preserves_original_guard_deadlines(
    tmp_path: Path, clock: SimpleNamespace
) -> None:
    """A healthy busy writer cannot trigger early shutdown or extend the cutoff."""
    initialize_pilot_budget(tmp_path)
    start_session(
        tmp_path, "drill", "i-12345678", "setup", clock.time(), requested_seconds=600
    )
    actions = Actions(clock)
    with ledger_lock(tmp_path):
        pilot_guard.run_guard(tmp_path, "drill", actions)
    assert actions.events == [("save", 300), ("poweroff", 480)]
    assert inspect_budget(tmp_path).consumed_seconds == 480


def test_missing_ledger_stops_without_initializing(
    tmp_path: Path, clock: SimpleNamespace
) -> None:
    """The armed host guard fails safe if its mounted accounting cannot be read."""
    actions = Actions(clock)
    directory = tmp_path / "missing"
    pilot_guard.run_guard(directory, "drill", actions)
    assert actions.events == [("save", 0), ("poweroff", 0)]
    assert not directory.exists()


def test_process_commands_are_bounded_and_use_no_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock every external action, including poweroff, while checking exact argv."""
    commands = []

    def run(command: list[str], **arguments: Any) -> subprocess.CompletedProcess[bytes]:
        """Capture calls without launching or terminating any process."""
        assert arguments == {"check": True, "timeout": 10, "capture_output": True}
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    actions = HostStopActions("stegolab-pilot", execute=True)
    actions.request_checkpoint()
    actions.poweroff()
    assert commands == [
        ["docker", "kill", "--signal=TERM", "stegolab-pilot"],
        ["systemctl", "poweroff"],
    ]
    actions = HostStopActions("stegolab-pilot")
    actions.request_checkpoint()
    actions.poweroff()
    assert len(commands) == 2
