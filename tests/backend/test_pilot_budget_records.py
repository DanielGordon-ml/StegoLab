"""Attested past sessions, stage transfers and per-stage totals."""

from pathlib import Path

import pytest
from pilot_budget_fixtures import EVIDENCE, LOST_START, LOST_STOP, patch_clocks, totals
from test_pilot_budget import Clock

from backend_service.failures import ApplicationFailure
from backend_service.pilot_budget import confirm_stopped, inspect_budget, start_session
from backend_service.pilot_budget_records import (
    attest_closed_session,
    transfer_allocation,
)
from backend_service.pilot_budget_storage import initialize_pilot_budget, load_ledger
from schemas.pilot_budget import PilotStage


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Clock:
    """Use the shared fake clocks for every test in this module."""
    return patch_clocks(monkeypatch, tmp_path)


def test_attested_hour_is_charged_to_setup_and_shown_per_stage(
    tmp_path: Path, clock: Clock
) -> None:
    """Record the 2026-09-24 hour from evidence and see it in the stage totals."""
    clock.seconds = LOST_STOP + 86_400 - 1_000_000.0
    initialize_pilot_budget(tmp_path)
    session = attest_closed_session(
        tmp_path,
        "lost_hour",
        "i-0ccaee67f0acaa574",
        "setup",
        LOST_START,
        LOST_STOP,
        EVIDENCE,
    )
    assert session.record_kind == "attested"
    assert session.stopped_at == LOST_STOP and session.evidence == EVIDENCE
    summary = inspect_budget(tmp_path)
    assert summary.consumed_seconds == 3629 and summary.remaining_seconds == 82_771
    assert summary.active_session_identifier is None
    assert totals(tmp_path)["setup"] == (7200, 3629, 3571)
    assert totals(tmp_path)["baseline"] == (50_400, 0, 50_400)
    saved = load_ledger(tmp_path)
    assert saved.sessions[0].stop_confirmation == EVIDENCE[:256]
    drill = start_session(
        tmp_path, "drill", "i-0ccaee67f0acaa574", "setup", clock.time()
    )
    assert drill.deadline_at - drill.started_at == 3571
    confirm_stopped(tmp_path, "drill", "operator observed EC2 stopped")


def test_attestation_refuses_bad_times_overlaps_and_over_allocation(
    tmp_path: Path, clock: Clock
) -> None:
    """Never let an attested record rewrite history or exceed a stage."""
    clock.seconds = 100_000
    initialize_pilot_budget(tmp_path)
    instance = "i-12345678"
    for started, stopped, message in (
        (clock.time() - 10, clock.time() + 10, "past times"),
        (2000.0, 1000.0, "past times"),
        (1000.0, 1000.0, "past times"),
        (1000.0, 1000.0 + 7201, "remaining allocation"),
    ):
        with pytest.raises((ValueError, ApplicationFailure), match=message):
            attest_closed_session(
                tmp_path, "bad", instance, "setup", started, stopped, "evidence"
            )
    attest_closed_session(
        tmp_path, "first", instance, "setup", 1_000_000.0, 1_001_000.0, "evidence"
    )
    with pytest.raises(ApplicationFailure, match="new session identifier"):
        attest_closed_session(
            tmp_path, "first", instance, "setup", 1_002_000.0, 1_003_000.0, "e"
        )
    with pytest.raises(ApplicationFailure, match="overlap"):
        attest_closed_session(
            tmp_path, "second", instance, "setup", 1_000_500.0, 1_001_500.0, "e"
        )
    earlier = attest_closed_session(
        tmp_path, "earlier", instance, "baseline", 900_000.0, 901_000.0, "e"
    )
    assert [item.session_identifier for item in load_ledger(tmp_path).sessions] == [
        "earlier",
        "first",
    ]
    assert earlier.stage == "baseline"
    start_session(tmp_path, "open", instance, "setup", clock.time())
    with pytest.raises(ApplicationFailure, match="stopped first"):
        attest_closed_session(
            tmp_path, "late", instance, "setup", 1_010_000.0, 1_011_000.0, "e"
        )


def test_transfers_move_unused_seconds_within_the_rules(
    tmp_path: Path, clock: Clock
) -> None:
    """Borrow at most one hour from a stage that still has the seconds unused."""
    clock.seconds = 100_000
    initialize_pilot_budget(tmp_path)
    attest_closed_session(
        tmp_path, "lost_hour", "i-12345678", "setup", 1_000_000.0, 1_003_629.0, "e"
    )
    transfer = transfer_allocation(
        tmp_path, "ablation", "setup", 3600, "Readiness drill retry after the lost hour"
    )
    assert transfer.recorded_at == clock.time()
    assert totals(tmp_path)["setup"] == (10_800, 3629, 7171)
    assert totals(tmp_path)["ablation"] == (10_800, 0, 10_800)
    assert inspect_budget(tmp_path).remaining_seconds == 82_771
    refusals: list[tuple[PilotStage, PilotStage, float, str]] = [
        ("ablation", "ablation", 100, "different stages"),
        ("ablation", "setup", 3601, "at a time"),
        ("ablation", "setup", 0, "at a time"),
        ("evaluation", "setup", float("inf"), "at a time"),
    ]
    for from_stage, to_stage, seconds, message in refusals:
        with pytest.raises(ValueError, match=message):
            transfer_allocation(tmp_path, from_stage, to_stage, seconds, "reason")
    for _ in range(3):
        transfer_allocation(tmp_path, "ablation", "evaluation", 3600, "drain")
    with pytest.raises(ApplicationFailure, match="fewer unused seconds"):
        transfer_allocation(tmp_path, "ablation", "evaluation", 1, "too much")
    start_session(tmp_path, "open", "i-12345678", "evaluation", clock.time())
    with pytest.raises(ApplicationFailure, match="fewer unused seconds"):
        transfer_allocation(tmp_path, "evaluation", "baseline", 3600, "reserved")
    assert len(load_ledger(tmp_path).stage_transfers) == 4


def test_closed_sessions_are_charged_by_used_time_for_transfers_and_attestations(
    tmp_path: Path, clock: Clock
) -> None:
    """A session that stopped early leaves its unused seconds free to move or fill."""
    clock.seconds = 100_000
    initialize_pilot_budget(tmp_path)
    start_session(tmp_path, "drill", "i-12345678", "setup", clock.time())
    clock.seconds += 600
    confirm_stopped(tmp_path, "drill", "operator observed EC2 stopped")
    assert totals(tmp_path)["setup"] == (7200, 600, 6600)
    transfer_allocation(tmp_path, "setup", "baseline", 3600, "unused setup time")
    assert totals(tmp_path)["setup"] == (3600, 600, 3000)
    earlier = attest_closed_session(
        tmp_path, "before_drill", "i-12345678", "setup", 1_000.0, 2_000.0, "e"
    )
    assert [item.session_identifier for item in load_ledger(tmp_path).sessions] == [
        earlier.session_identifier,
        "drill",
    ]
    assert totals(tmp_path)["setup"] == (3600, 1600, 2000)
    with pytest.raises(ApplicationFailure, match="fewer unused seconds"):
        transfer_allocation(tmp_path, "setup", "baseline", 2001, "too much")
    start_session(
        tmp_path, "overrun", "i-12345678", "setup", clock.time(), requested_seconds=301
    )
    clock.seconds += 2000
    confirm_stopped(tmp_path, "overrun", "operator observed EC2 stopped")
    assert totals(tmp_path)["setup"] == (3600, 3600, 0)
    with pytest.raises(ApplicationFailure, match="fewer unused seconds"):
        transfer_allocation(tmp_path, "setup", "ablation", 1, "spent by the overrun")
    assert load_ledger(tmp_path).sessions[-1].stopped_at == clock.time()
