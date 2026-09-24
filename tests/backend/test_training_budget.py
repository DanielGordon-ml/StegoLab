"""Persistent proof limits, restart accounting, exclusive use, and safe failures."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend_service import training_budget
from backend_service.failures import ApplicationFailure
from backend_service.training_budget import ProofBudget
from backend_service.training_budget_records import ActiveBudget, BudgetLedger
from backend_service.training_checkpoint_files import atomic_json


@dataclass
class Clock:
    """Advance simulated wall time without waiting through real proof budgets."""

    seconds: float = 0.0

    def monotonic(self) -> float:
        """Return a deterministic elapsed-time reading."""
        return self.seconds

    def time(self) -> float:
        """Provide a separate positive wall-clock timestamp for saved provenance."""
        return 1_000_000.0 + self.seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Clock:
    """Use fake time and ample disk without changing unrelated global clocks."""
    clock = Clock()
    monkeypatch.setattr(training_budget, "time", clock)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: usage._replace(total=200 * 1024**3, free=150 * 1024**3),
    )
    return clock


def test_resume_charges_all_elapsed_time_and_preserves_reserve(
    tmp_path: Path,
    clock: Clock,
) -> None:
    """Training and later evaluation time share the same persistent allowance."""
    with ProofBudget(tmp_path, "first") as budget:
        assert budget.remaining_seconds == 7200
        assert budget.training_remaining_seconds == 6000
        clock.seconds += 100
    assert budget.elapsed_seconds == 100
    with ProofBudget(tmp_path, "first", resume=True) as resumed:
        assert resumed.remaining_seconds == 7100
        clock.seconds += 5900
        assert resumed.training_remaining_seconds == 0
        assert resumed.remaining_seconds == 1200
        with pytest.raises(ApplicationFailure):
            resumed.check(training=True)
        resumed.check()
        clock.seconds += 600
    ledger = BudgetLedger.model_validate_json((tmp_path / "ledger.json").read_bytes())
    assert ledger.consumed_seconds == 6600
    assert ledger.experiments[0].consumed_seconds == 6600
    assert ledger.active is None


def test_limited_experiment_slots_and_explicit_resume(
    tmp_path: Path,
    clock: Clock,
) -> None:
    """Reusing a name needs resume and changing names cannot create a third slot."""
    for identifier in ("first", "second"):
        with ProofBudget(tmp_path, identifier):
            clock.seconds += 10
    for identifier, resume in (("third", False), ("first", False), ("missing", True)):
        with pytest.raises(ApplicationFailure):
            with ProofBudget(tmp_path, identifier, resume=resume):
                pytest.fail("Invalid experiment ownership must be refused")
    with ProofBudget(tmp_path, "second", resume=True) as resumed:
        assert resumed.remaining_seconds == 7190


def test_concurrent_operations_cannot_share_budget(
    tmp_path: Path, clock: Clock
) -> None:
    """The project-wide lock covers evaluation and exports as well as training."""
    with ProofBudget(tmp_path, "first"):
        with pytest.raises(ApplicationFailure, match="already running"):
            with ProofBudget(tmp_path, "second"):
                pytest.fail("A second operation must never own the ledger")
        clock.seconds += 3
    with ProofBudget(tmp_path, "second"):
        pass


def test_crash_charges_reservation_before_any_new_operation(
    tmp_path: Path,
    clock: Clock,
) -> None:
    """An abandoned persistent reservation spends its full unknown time allowance."""
    with ProofBudget(tmp_path, "first"):
        clock.seconds += 10
    ledger = BudgetLedger.model_validate_json((tmp_path / "ledger.json").read_bytes())
    ledger.active = ActiveBudget(
        experiment_identifier="first",
        started_at=clock.time(),
        reserved_seconds=7190.0,
    )
    atomic_json(tmp_path / "ledger.json", ledger.model_dump(mode="json"))
    with pytest.raises(ApplicationFailure, match="exhausted"):
        with ProofBudget(tmp_path, "first", resume=True):
            pytest.fail("Crash recovery cannot reset the spent reservation")
    with ProofBudget(tmp_path, "second") as budget:
        assert budget.remaining_seconds == 7200
        assert budget.ledger.consumed_seconds == 7200
        clock.seconds += 7200
    with pytest.raises(ApplicationFailure, match="exhausted"):
        with ProofBudget(tmp_path, "second", resume=True):
            pytest.fail("The four-hour project envelope must remain exhausted")


def test_missing_or_corrupt_initialized_ledger_never_resets(
    tmp_path: Path,
    clock: Clock,
) -> None:
    """Lost accounting state fails closed instead of granting new proof time."""
    with ProofBudget(tmp_path, "first"):
        clock.seconds += 2
    saved = (tmp_path / "ledger.json").read_bytes()
    for payload in (None, b"{broken"):
        if payload is None:
            (tmp_path / "ledger.json").unlink()
        else:
            (tmp_path / "ledger.json").write_bytes(payload)
        with pytest.raises(ApplicationFailure):
            with ProofBudget(tmp_path, "second"):
                pytest.fail("Missing or broken accounting must stop further proof work")
    (tmp_path / "ledger.json").write_bytes(saved)
    with ProofBudget(tmp_path, "first", resume=True) as resumed:
        assert resumed.remaining_seconds == 7198


def test_preflight_failure_can_retry_original_empty_checkpoint_directory(
    tmp_path: Path,
    clock: Clock,
) -> None:
    """A corrected startup request reuses its spent slot without fresh time."""
    ledger_root = tmp_path / "state" / "cpu_proof"
    checkpoint_directory = tmp_path / "checkpoints" / "first"
    with pytest.raises(ApplicationFailure, match="dataset unavailable"):
        with ProofBudget(
            ledger_root, "first", checkpoint_directory=checkpoint_directory
        ):
            clock.seconds += 12
            raise ApplicationFailure("dataset_invalid", "dataset unavailable")
    with ProofBudget(
        ledger_root, "first", checkpoint_directory=checkpoint_directory
    ) as retry:
        assert retry.remaining_seconds == 7188
        assert len(retry.ledger.experiments) == 1
        clock.seconds += 3
    ledger = BudgetLedger.model_validate_json(
        (ledger_root / "ledger.json").read_bytes()
    )
    assert ledger.consumed_seconds == 15
    assert ledger.experiments[0].consumed_seconds == 15


@pytest.mark.parametrize("entry", ["index.json", "checkpoint_1_completed", ".pending"])
def test_existing_checkpoint_or_uncertain_storage_requires_explicit_resume(
    tmp_path: Path,
    clock: Clock,
    entry: str,
) -> None:
    """Published, malformed, or ambiguous state never becomes an implicit restart."""
    ledger_root = tmp_path / "state" / "cpu_proof"
    checkpoint_directory = tmp_path / "checkpoints" / "first"
    with ProofBudget(ledger_root, "first"):
        clock.seconds += 10
    checkpoint_directory.mkdir(parents=True)
    if entry == "index.json":
        (checkpoint_directory / entry).write_bytes(b"{corrupt")
    else:
        (checkpoint_directory / entry).mkdir()
    with pytest.raises(ApplicationFailure, match="Use resume"):
        with ProofBudget(
            ledger_root, "first", checkpoint_directory=checkpoint_directory
        ):
            pytest.fail("Existing state must require checkpoint resume validation")
    with ProofBudget(ledger_root, "first", resume=True) as resume:
        assert resume.remaining_seconds == 7190


def test_startup_retry_rejects_other_experiment_directory_and_symlink(
    tmp_path: Path,
    clock: Clock,
) -> None:
    """Another empty path or redirected storage cannot hide a saved checkpoint."""
    ledger_root = tmp_path / "state" / "cpu_proof"
    checkpoint_directory = tmp_path / "checkpoints" / "first"
    with ProofBudget(ledger_root, "first"):
        clock.seconds += 1
    for wrong_path in (
        tmp_path / "checkpoints" / "second",
        tmp_path / "unrelated",
    ):
        with pytest.raises(ApplicationFailure, match="original checkpoint directory"):
            with ProofBudget(ledger_root, "first", checkpoint_directory=wrong_path):
                pytest.fail("Only this experiment's storage may authorize a retry")
    checkpoint_directory.parent.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    checkpoint_directory.symlink_to(other, target_is_directory=True)
    with pytest.raises(ApplicationFailure, match="unsafe"):
        with ProofBudget(
            ledger_root, "first", checkpoint_directory=checkpoint_directory
        ):
            pytest.fail("Redirected checkpoint storage cannot authorize a retry")
