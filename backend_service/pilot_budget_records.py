"""Operator records for instance time that ran without a session, and transfers."""

import bisect
import math
import time
from pathlib import Path

from backend_service.pilot_budget import consumed_seconds
from backend_service.pilot_budget_storage import (
    budget_failure,
    ledger_lock,
    load_ledger,
    save_ledger,
)
from schemas.pilot_budget import (
    MAXIMUM_TRANSFER_SECONDS,
    STAGE_SECONDS,
    PilotBudgetLedger,
    PilotSession,
    PilotStage,
    StageTransfer,
)


def _reserved_seconds(ledger: PilotBudgetLedger, stage: PilotStage) -> float:
    """Count closed sessions by elapsed time and an open session by its allowance."""
    return sum(
        (session.stopped_at - session.started_at)
        if session.stopped_at is not None
        else (session.deadline_at - session.started_at)
        for session in ledger.sessions
        if session.stage == stage
    )


def attest_closed_session(
    directory: Path,
    session_identifier: str,
    instance_identifier: str,
    stage: PilotStage,
    started_at: float,
    stopped_at: float,
    evidence: str,
) -> PilotSession:
    """Charge a past instance run that had no session, from independent evidence."""
    # Argument checks happen before the lock, whose failures are storage failures.
    now = time.time()
    if stage not in STAGE_SECONDS:
        raise ValueError("Choose a supported pilot stage.")
    if not (
        math.isfinite(started_at)
        and math.isfinite(stopped_at)
        and 0 <= started_at < stopped_at <= now
    ):
        raise ValueError("Start and stop must be real past times, start first.")
    with ledger_lock(directory):
        ledger = load_ledger(directory)
        if ledger.sessions and ledger.sessions[-1].stopped_at is None:
            raise budget_failure("Confirm the active instance is stopped first.")
        if any(
            item.session_identifier == session_identifier for item in ledger.sessions
        ):
            raise budget_failure(
                "Use a new session identifier; saved sessions are final."
            )
        if any(
            item.stopped_at is not None
            and started_at < item.stopped_at
            and item.started_at < stopped_at
            for item in ledger.sessions
        ):
            raise budget_failure("The attested times overlap a recorded session.")
        elapsed = stopped_at - started_at
        remaining = min(
            ledger.maximum_total_seconds - consumed_seconds(ledger, now),
            ledger.effective_allocations()[stage] - _reserved_seconds(ledger, stage),
        )
        if elapsed > remaining + 1e-6:
            raise budget_failure(
                "The attested session exceeds the remaining allocation of its stage."
            )
        session = PilotSession(
            session_identifier=session_identifier,
            instance_identifier=instance_identifier,
            stage=stage,
            record_kind="attested",
            started_at=started_at,
            deadline_at=stopped_at,
            checkpoint_at=stopped_at - ledger.checkpoint_reserve_seconds,
            poweroff_at=stopped_at - ledger.shutdown_reserve_seconds,
            last_observed_at=stopped_at,
            stopped_at=stopped_at,
            stop_confirmation=evidence[:256],
            evidence=evidence,
        )
        # Keep sessions in time order so the ledger's overlap rules stay simple.
        position = bisect.bisect_left(
            [item.started_at for item in ledger.sessions], started_at
        )
        ledger.sessions.insert(position, session)
        save_ledger(directory, ledger)
        return session


def transfer_allocation(
    directory: Path,
    from_stage: PilotStage,
    to_stage: PilotStage,
    seconds: float,
    reason: str,
) -> StageTransfer:
    """Move unused seconds between stages; the total never changes."""
    if from_stage not in STAGE_SECONDS or to_stage not in STAGE_SECONDS:
        raise ValueError("Choose supported pilot stages.")
    if from_stage == to_stage:
        raise ValueError("Choose two different stages for a transfer.")
    if not math.isfinite(seconds) or not 0 < seconds <= MAXIMUM_TRANSFER_SECONDS:
        raise ValueError(
            f"Transfer between 1 and {MAXIMUM_TRANSFER_SECONDS} seconds at a time."
        )
    with ledger_lock(directory):
        ledger = load_ledger(directory)
        available = ledger.effective_allocations()[from_stage] - _reserved_seconds(
            ledger, from_stage
        )
        if seconds > available + 1e-6:
            raise budget_failure(
                "The source stage has fewer unused seconds than the transfer."
            )
        transfer = StageTransfer(
            from_stage=from_stage,
            to_stage=to_stage,
            seconds=seconds,
            reason=reason,
            recorded_at=time.time(),
        )
        ledger.stage_transfers.append(transfer)
        save_ledger(directory, ledger)
        return transfer
