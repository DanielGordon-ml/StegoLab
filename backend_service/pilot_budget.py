"""Operator-managed instance time and exclusive leases for explicit GPU work."""

import fcntl
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from backend_service.pilot_budget_storage import (
    budget_failure,
    ledger_lock,
    load_ledger,
    save_ledger,
)
from schemas.pilot_budget import (
    STAGE_SECONDS,
    PilotBudgetLedger,
    PilotBudgetSummary,
    PilotSession,
    PilotStage,
    PilotStageTotal,
)


def consumed_seconds(
    ledger: PilotBudgetLedger, observed_at: float, stage: PilotStage | None = None
) -> float:
    """Charge startup, idle, all jobs, shutdown, and any overrun without capping it."""
    return sum(
        (session.stopped_at or max(observed_at, session.last_observed_at))
        - session.started_at
        for session in ledger.sessions
        if stage is None or session.stage == stage
    )


def stage_totals(
    ledger: PilotBudgetLedger, observed_at: float
) -> list[PilotStageTotal]:
    """Break spending down per stage against the allocations after transfers."""
    totals: list[PilotStageTotal] = []
    for stage, allocated in ledger.effective_allocations().items():
        spent = consumed_seconds(ledger, observed_at, stage)
        totals.append(
            PilotStageTotal(
                stage=stage,
                allocated_seconds=allocated,
                consumed_seconds=spent,
                remaining_seconds=max(0.0, allocated - spent),
            )
        )
    return totals


def inspect_budget(directory: Path) -> PilotBudgetSummary:
    """Inspect without creating storage; damaged initialized state fails closed."""
    try:
        if directory.is_symlink():
            raise ValueError
        if not directory.exists() or (
            directory.is_dir() and not any(directory.iterdir())
        ):
            return PilotBudgetSummary(
                initialized=False, consumed_seconds=0.0, remaining_seconds=86400.0
            )
        ledger = load_ledger(directory)
        spent = consumed_seconds(ledger, time.time())
        active = next(
            (
                item.session_identifier
                for item in ledger.sessions
                if item.stopped_at is None
            ),
            None,
        )
        return PilotBudgetSummary(
            initialized=True,
            consumed_seconds=spent,
            remaining_seconds=max(0.0, ledger.maximum_total_seconds - spent),
            active_session_identifier=active,
            stage_totals=stage_totals(ledger, time.time()),
        )
    except (OSError, ValueError):
        raise budget_failure(
            "The saved pilot budget is invalid; restore its backup."
        ) from None


def _active(ledger: PilotBudgetLedger, identifier: str) -> PilotSession:
    """Require the exact active session; interrupted jobs never open new sessions."""
    if (
        not ledger.sessions
        or ledger.sessions[-1].session_identifier != identifier
        or ledger.sessions[-1].stopped_at is not None
    ):
        raise budget_failure("No matching active instance session. Check the ledger.")
    return ledger.sessions[-1]


def start_session(
    directory: Path,
    session_identifier: str,
    instance_identifier: str,
    stage: PilotStage,
    started_at: float,
    *,
    requested_seconds: float | None = None,
) -> PilotSession:
    """Register an operator-launched instance, including already elapsed startup."""
    with ledger_lock(directory):
        ledger = load_ledger(directory)
        now = time.time()
        if stage not in STAGE_SECONDS:
            raise ValueError("Choose a supported pilot stage.")
        if not math.isfinite(started_at) or not 0 <= started_at <= now:
            raise ValueError("Startup time must be a real past timestamp.")
        if ledger.sessions and ledger.sessions[-1].stopped_at is None:
            raise budget_failure("Confirm the active instance is stopped first.")
        if any(
            item.session_identifier == session_identifier for item in ledger.sessions
        ):
            raise budget_failure(
                "Use a new session identifier; saved sessions are final."
            )
        remaining = min(
            ledger.maximum_total_seconds - consumed_seconds(ledger, now),
            ledger.effective_allocations()[stage]
            - consumed_seconds(ledger, now, stage),
        )
        allowance = remaining if requested_seconds is None else requested_seconds
        if not math.isfinite(allowance) or not 300 < allowance <= remaining:
            raise budget_failure("The requested session exceeds its remaining budget.")
        session = PilotSession(
            session_identifier=session_identifier,
            instance_identifier=instance_identifier,
            stage=stage,
            started_at=started_at,
            deadline_at=started_at + allowance,
            checkpoint_at=started_at + allowance - ledger.checkpoint_reserve_seconds,
            poweroff_at=started_at + allowance - ledger.shutdown_reserve_seconds,
            last_observed_at=now,
        )
        ledger.sessions.append(session)
        save_ledger(directory, ledger)
        return session


def read_session(directory: Path, session_identifier: str) -> PilotSession:
    """Read an atomically published session without racing on the writer lock."""
    try:
        if directory.is_symlink():
            raise ValueError
        session = _active(load_ledger(directory), session_identifier)
        if time.time() < session.last_observed_at:
            raise budget_failure("The instance clock moved backwards. Save and stop.")
        return session
    except (OSError, ValueError):
        raise budget_failure(
            "The saved instance session is invalid. Save and stop."
        ) from None


def observe_session(directory: Path, session_identifier: str) -> PilotSession:
    """Persist a conservative clock watermark without extending any deadline."""
    with ledger_lock(directory):
        ledger = load_ledger(directory)
        session = _active(ledger, session_identifier)
        now = time.time()
        if now < session.last_observed_at:
            raise budget_failure("The instance clock moved backwards. Save and stop.")
        session.last_observed_at = now
        save_ledger(directory, ledger)
        return session


def confirm_stopped(
    directory: Path, session_identifier: str, confirmation: str
) -> PilotBudgetLedger:
    """Close only after the operator observes EC2 stopped; charge until that check."""
    with ledger_lock(directory):
        ledger = load_ledger(directory)
        session = _active(ledger, session_identifier)
        now = time.time()
        session.stopped_at = (
            max(session.last_observed_at, session.deadline_at)
            if now < session.last_observed_at
            else now
        )
        session.stop_confirmation = confirmation
        save_ledger(directory, ledger)
        return ledger


@dataclass(frozen=True)
class PilotDeadline:
    """Anchor an existing wall deadline to monotonic time without granting time."""

    deadline_unix_seconds: float
    checkpoint_unix_seconds: float
    monotonic_deadline: float
    checkpoint_monotonic_deadline: float

    @property
    def remaining_seconds(self) -> float:
        """Return time until forced shutdown, including the save reserve."""
        return max(0.0, self.monotonic_deadline - time.monotonic())

    @property
    def checkpoint_remaining_seconds(self) -> float:
        """Return work time before safe checkpointing must begin."""
        return max(0.0, self.checkpoint_monotonic_deadline - time.monotonic())


def read_pilot_deadline(directory: Path, session_identifier: str) -> PilotDeadline:
    """Validate an active session and convert its existing deadlines, never start it."""
    session = observe_session(directory, session_identifier)
    now = max(time.time(), session.last_observed_at)
    monotonic = time.monotonic()
    if session.checkpoint_at <= now:
        raise budget_failure("The pilot save reserve has begun. Save and stop work.")
    return PilotDeadline(
        deadline_unix_seconds=session.deadline_at,
        checkpoint_unix_seconds=session.checkpoint_at,
        monotonic_deadline=monotonic + session.deadline_at - now,
        checkpoint_monotonic_deadline=monotonic + session.checkpoint_at - now,
    )


class PilotOperation:
    """Keep one train/evaluate/export operation in an already running GPU session."""

    def __init__(self, directory: Path, session_identifier: str) -> None:
        """Select operator accounting without creating files or claiming a GPU."""
        self.directory = directory
        self.session_identifier = session_identifier
        self._descriptor: int | None = None

    def __enter__(self) -> PilotDeadline:
        """Acquire the operation lock before validating its immutable deadline."""
        try:
            if self.directory.is_symlink() or not self.directory.is_dir():
                raise ValueError
            self._descriptor = os.open(
                self.directory / ".operation.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
            )
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return read_pilot_deadline(self.directory, self.session_identifier)
        except BlockingIOError:
            self._release()
            raise budget_failure("Another GPU operation is already running.") from None
        except (OSError, ValueError):
            self._release()
            raise budget_failure("The pilot operation lock is unavailable.") from None
        except BaseException:
            self._release()
            raise

    def _release(self) -> None:
        """Release only operation ownership; the instance remains chargeable."""
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave session accounting active even if the job crashes or completes."""
        self._release()
