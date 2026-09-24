"""Instance-wide accounting, conservative restart handling, and exclusive GPU work."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend_service import pilot_budget
from backend_service.failures import ApplicationFailure
from backend_service.pilot_budget import (
    PilotOperation,
    confirm_stopped,
    inspect_budget,
    observe_session,
    start_session,
)
from backend_service.pilot_budget_storage import initialize_pilot_budget
from schemas.pilot_budget import PilotBudgetLedger


@dataclass
class Clock:
    """Provide distinct wall and monotonic clocks without real waiting."""

    seconds: float = 0.0
    wall_offset: float = 0.0

    def time(self) -> float:
        """Return a realistic positive epoch including deliberate wall changes."""
        return 1_000_000.0 + self.seconds + self.wall_offset

    def monotonic(self) -> float:
        """Advance elapsed time independently from wall-clock adjustments."""
        return self.seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Clock:
    """Patch this module's clocks and provide space for tiny ledger fixtures."""
    value = Clock()
    monkeypatch.setattr(pilot_budget, "time", value)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: usage._replace(total=200 * 1024**3, free=150 * 1024**3),
    )
    return value


def test_inspection_is_read_only_and_never_initializes(
    tmp_path: Path, clock: Clock
) -> None:
    """Preflight reports an unused missing ledger without granting a running session."""
    directory = tmp_path / "missing"
    summary = inspect_budget(directory)
    assert not summary.initialized and summary.remaining_seconds == 86400
    assert not directory.exists()
    with pytest.raises(ApplicationFailure):
        with PilotOperation(directory, "first"):
            pytest.fail("GPU work needs an operator-created session")
    assert not directory.exists()


def test_all_instance_time_counts_until_confirmed_stopped(
    tmp_path: Path, clock: Clock
) -> None:
    """Startup, idle and completed/failed jobs do not reset or close instance time."""
    initialize_pilot_budget(tmp_path)
    clock.seconds = 30
    start_session(tmp_path, "first", "i-12345678", "setup", 1_000_000.0)
    with PilotOperation(tmp_path, "first") as deadline:
        assert deadline.monotonic_deadline == 7200
        assert deadline.checkpoint_monotonic_deadline == 6900
        clock.seconds = 100
    assert inspect_budget(tmp_path).active_session_identifier == "first"
    clock.seconds = 300
    with pytest.raises(RuntimeError):
        with PilotOperation(tmp_path, "first"):
            raise RuntimeError("simulated application crash")
    clock.seconds = 600
    ledger = confirm_stopped(tmp_path, "first", "operator observed EC2 stopped")
    assert ledger.sessions[0].stopped_at == clock.time()
    assert inspect_budget(tmp_path).consumed_seconds == 600
    start_session(tmp_path, "second", "i-12345678", "setup", clock.time())
    with PilotOperation(tmp_path, "second") as deadline:
        assert deadline.remaining_seconds == 6600


def test_sessions_and_gpu_operations_are_exclusive(
    tmp_path: Path, clock: Clock
) -> None:
    """The separate ledger lock lets the host observe while a single GPU job runs."""
    initialize_pilot_budget(tmp_path)
    start_session(tmp_path, "first", "i-12345678", "setup", clock.time())
    with PilotOperation(tmp_path, "first"):
        with pytest.raises(ApplicationFailure, match="already running"):
            with PilotOperation(tmp_path, "first"):
                pytest.fail("Concurrent model work must be rejected")
        clock.seconds = 10
        assert observe_session(tmp_path, "first").last_observed_at == clock.time()
        with pytest.raises(ApplicationFailure, match="stopped first"):
            start_session(tmp_path, "second", "i-12345679", "baseline", clock.time())
    with PilotOperation(tmp_path, "first"):
        pass


def test_save_reserve_and_crash_restart_do_not_extend_deadline(
    tmp_path: Path, clock: Clock
) -> None:
    """A new process recovers the same absolute deadline from an active session."""
    initialize_pilot_budget(tmp_path)
    start_session(tmp_path, "first", "i-12345678", "setup", clock.time())
    clock.seconds = 6820
    with PilotOperation(tmp_path, "first") as lease:
        assert lease.checkpoint_remaining_seconds == 80
        clock.wall_offset = -500
        clock.seconds = 6900
        assert lease.checkpoint_remaining_seconds == 0
    with pytest.raises(ApplicationFailure, match="backwards"):
        with PilotOperation(tmp_path, "first"):
            pytest.fail("A clock rollback cannot grant more time")
    ledger = confirm_stopped(tmp_path, "first", "stopped after clock fault")
    assert ledger.sessions[0].stopped_at == 1_007_200


def test_setup_cap_and_overruns_are_not_forgiven(tmp_path: Path, clock: Clock) -> None:
    """Spent setup cannot restart and shutdown overhead may exceed the deadline."""
    initialize_pilot_budget(tmp_path)
    with pytest.raises(ApplicationFailure):
        start_session(
            tmp_path,
            "first",
            "i-12345678",
            "setup",
            clock.time(),
            requested_seconds=7201,
        )
    start_session(tmp_path, "first", "i-12345678", "setup", clock.time())
    clock.seconds = 7205
    with pytest.raises(ApplicationFailure, match="save reserve"):
        with PilotOperation(tmp_path, "first"):
            pytest.fail("Expired sessions cannot accept model work")
    confirm_stopped(tmp_path, "first", "operator observed stopped")
    assert inspect_budget(tmp_path).consumed_seconds == 7205
    with pytest.raises(ApplicationFailure):
        start_session(tmp_path, "second", "i-12345678", "setup", clock.time())
    start_session(tmp_path, "baseline", "i-12345678", "baseline", clock.time())
    clock.seconds += 50400
    confirm_stopped(tmp_path, "baseline", "operator observed stopped")
    start_session(tmp_path, "ablation", "i-12345678", "ablation", clock.time())
    clock.seconds += 14400
    confirm_stopped(tmp_path, "ablation", "operator observed stopped")
    last = start_session(
        tmp_path, "evaluation", "i-12345678", "evaluation", clock.time()
    )
    assert last.deadline_at - last.started_at == 14395
    clock.seconds += 14395
    confirm_stopped(tmp_path, "evaluation", "operator observed stopped")
    assert inspect_budget(tmp_path).remaining_seconds == 0


@pytest.mark.parametrize("damage", ["missing", "corrupt", "marker", "changed_cap"])
def test_invalid_saved_accounting_never_resets(
    tmp_path: Path, clock: Clock, damage: str
) -> None:
    """Lost state requires restoration rather than a fresh twenty-four hours."""
    initialize_pilot_budget(tmp_path)
    ledger = tmp_path / "ledger.json"
    if damage == "missing":
        ledger.unlink()
    elif damage == "corrupt":
        ledger.write_text("broken")
    elif damage == "marker":
        (tmp_path / ".initialized.json").unlink()
    else:
        ledger.write_text(ledger.read_text().replace("86400", "86401"))
    with pytest.raises(ApplicationFailure):
        inspect_budget(tmp_path)
    with pytest.raises(ApplicationFailure):
        initialize_pilot_budget(tmp_path)


def test_fixed_ledger_numbers_reject_numeric_lookalikes() -> None:
    """Strict literal settings cannot be changed through equal bool/float values."""
    for value in (True, 1.0):
        with pytest.raises(ValueError):
            PilotBudgetLedger.model_validate({"schema_version": value})
