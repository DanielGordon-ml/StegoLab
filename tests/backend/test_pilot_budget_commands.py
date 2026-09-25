"""Ledger command output: records on success, named reasons on failure."""

from pathlib import Path

import pytest
from pilot_budget_fixtures import EVIDENCE, LOST_START, LOST_STOP, patch_clocks
from test_pilot_budget import Clock

from backend_service.pilot_budget import inspect_budget
from scripts.pilot_budget import main


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Clock:
    """Use the shared fake clocks for every test in this module."""
    return patch_clocks(monkeypatch, tmp_path)


def test_command_line_prints_records_and_failure_codes(
    tmp_path: Path, clock: Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Operators see the saved record on success and a named reason on failure."""
    clock.seconds = LOST_STOP + 86_400 - 1_000_000.0
    ledger = ["--ledger", str(tmp_path / "ledger")]
    assert main([*ledger, "initialize"]) == 0
    assert (
        main(
            [
                *ledger,
                "attest-closed",
                "--session",
                "lost_hour",
                "--instance",
                "i-0ccaee67f0acaa574",
                "--stage",
                "setup",
                "--started-at",
                str(LOST_START),
                "--stopped-at",
                str(LOST_STOP),
                "--evidence",
                EVIDENCE,
            ]
        )
        == 0
    )
    assert '"record_kind": "attested"' in capsys.readouterr().out
    assert main([*ledger, "inspect"]) == 0
    assert '"remaining_seconds": 3571.0' in capsys.readouterr().out
    assert (
        main(
            [
                *ledger,
                "transfer",
                "--from-stage",
                "ablation",
                "--to-stage",
                "setup",
                "--seconds",
                "4000",
                "--reason",
                "too much",
            ]
        )
        == 1
    )
    assert capsys.readouterr().err.startswith("pilot_accounting_invalid:")
    assert (
        main([*ledger, "confirm-stopped", "--session", "x", "--confirmation", "y"]) == 1
    )
    assert capsys.readouterr().err.startswith("pilot_budget_unavailable:")
    assert main(["--ledger", str(tmp_path / "missing"), "inspect"]) == 0


def test_operator_mistakes_name_the_mistake_not_the_ledger(
    tmp_path: Path, clock: Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Input errors and rule breaks never tell the operator to restore a backup."""
    clock.seconds = 100_000
    ledger = ["--ledger", str(tmp_path)]
    assert main([*ledger, "initialize"]) == 0
    capsys.readouterr()
    template = [
        *ledger,
        "attest-closed",
        "--session",
        "lost_hour",
        "--instance",
        "i-REPLACE",
        "--stage",
        "setup",
        "--started-at",
        "1000",
        "--stopped-at",
        "2000",
        "--evidence",
        "CloudTrail",
    ]
    assert main(template) == 1
    error = capsys.readouterr().err
    assert error.startswith("pilot_accounting_invalid:") and "backup" not in error
    assert (
        main(
            [
                *ledger,
                "start",
                "--session",
                "s",
                "--instance",
                "i-12345678",
                "--stage",
                "setup",
                "--started-at",
                str(clock.time() + 10),
            ]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert error.startswith("pilot_accounting_invalid:") and "backup" not in error
    assert inspect_budget(tmp_path).consumed_seconds == 0
    (tmp_path / "ledger.json").write_text("broken")
    assert main([*ledger, "inspect"]) == 1
    assert "Restore its verified backup" in capsys.readouterr().err
