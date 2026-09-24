"""Read-only CPU accounting includes active reservations and damaged ledger states."""

from pathlib import Path

from backend_service.training_budget_records import (
    ActiveBudget,
    BudgetLedger,
    ExperimentBudget,
)
from backend_service.workspace_eligibility import read_budget


def test_budget_read_does_not_initialize_or_reset_ledger(tmp_path: Path) -> None:
    """A fresh workspace reports defaults without spending or persisting time."""
    summary = read_budget(tmp_path)
    assert summary.remaining_seconds == 14400
    assert summary.remaining_experiments == 2
    assert not list(tmp_path.iterdir())
    directory = tmp_path / "state" / "cpu_proof"
    directory.mkdir(parents=True)
    (directory / ".initialized.json").write_text("{}")
    assert read_budget(tmp_path).blocked_reason is not None
    assert not (directory / "ledger.json").exists()


def test_budget_reservation_is_conservative_and_original_bytes_survive(
    tmp_path: Path,
) -> None:
    """Reading active accounting does not reconcile crashes or grant new allowance."""
    ledger = BudgetLedger(
        consumed_seconds=100.0,
        experiments=[
            ExperimentBudget(experiment_identifier="original", consumed_seconds=100.0)
        ],
        active=ActiveBudget(
            experiment_identifier="original", started_at=1.0, reserved_seconds=7100.0
        ),
    )
    path = tmp_path / "state" / "cpu_proof" / "ledger.json"
    path.parent.mkdir(parents=True)
    original = ledger.model_dump_json().encode()
    path.write_bytes(original)
    summary = read_budget(tmp_path)
    assert summary.remaining_seconds == 7200
    assert summary.remaining_experiments == 1
    assert summary.blocked_reason is not None
    assert path.read_bytes() == original
    path.write_bytes(b"broken ledger")
    assert read_budget(tmp_path).remaining_seconds == 0
    assert path.read_bytes() == b"broken ledger"
