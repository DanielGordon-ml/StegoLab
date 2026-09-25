"""Independent host deadline enforcement with injectable, testable stop actions."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend_service.failures import ApplicationFailure
from backend_service.pilot_budget import observe_session, read_session
from schemas.pilot_budget import PilotSession


class StopActions(Protocol):
    """Separate deadline decisions from process signals and host power controls."""

    def request_checkpoint(self) -> None:
        """Request a safe completed-step save from the selected training process."""
        ...

    def poweroff(self) -> None:
        """Request operating-system poweroff; EC2 must be configured to stop."""
        ...


@dataclass
class GuardDeadlines:
    """Keep conservative monotonic targets that may shorten but never extend."""

    checkpoint: float
    stop: float

    @classmethod
    def from_session(cls, session: PilotSession) -> "GuardDeadlines":
        """Anchor persisted epoch deadlines to this host process's monotonic clock."""
        now = max(time.time(), session.last_observed_at)
        monotonic = time.monotonic()
        return cls(
            checkpoint=monotonic + max(0.0, session.checkpoint_at - now),
            stop=monotonic + max(0.0, session.poweroff_at - now),
        )

    def shorten(self, session: PilotSession) -> None:
        """Apply forward wall-clock changes without extending original deadlines."""
        current = self.from_session(session)
        self.checkpoint = min(self.checkpoint, current.checkpoint)
        self.stop = min(self.stop, current.stop)


def run_guard(directory: Path, session_identifier: str, actions: StopActions) -> None:
    """Signal a safe save, then request poweroff even if the application is stuck."""
    checkpoint_requested = False
    try:
        session = read_session(directory, session_identifier)
        deadlines = GuardDeadlines.from_session(session)
        while True:
            now = time.monotonic()
            if now >= deadlines.checkpoint and not checkpoint_requested:
                checkpoint_requested = True
                try:
                    actions.request_checkpoint()
                except OSError:
                    pass
            if now >= deadlines.stop:
                break
            interval = 5.0
            try:
                session = observe_session(directory, session_identifier)
            except ApplicationFailure as failure:
                if failure.code != "pilot_budget_busy":
                    raise
                # Retry contention against the original monotonic cutoff.
                interval = 1.0
            else:
                deadlines.shorten(session)
            next_action = (
                deadlines.stop if checkpoint_requested else deadlines.checkpoint
            )
            time.sleep(min(interval, max(0.0, next_action - time.monotonic())))
    except (ApplicationFailure, OSError, ValueError) as failure:
        # An uncertain ledger/clock must not keep a paid instance running, and the
        # host journal should say why the guard stopped early.
        reason = (
            failure.message
            if isinstance(failure, ApplicationFailure)
            else type(failure).__name__
        )
        print(f"pilot_guard_stopping_early: {reason}", flush=True)
        if not checkpoint_requested:
            try:
                actions.request_checkpoint()
            except OSError:
                pass
    finally:
        # Only the operator confirms EC2 stopped; this call never closes accounting.
        actions.poweroff()
