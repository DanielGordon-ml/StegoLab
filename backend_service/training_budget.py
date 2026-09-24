"""Persistent experiment and wall-clock limits for the local CPU proof."""

import fcntl
import os
import time
from pathlib import Path
from types import TracebackType

from backend_service.failures import ApplicationFailure
from backend_service.training_budget_records import (
    ActiveBudget,
    BudgetLedger,
    ExperimentBudget,
)
from backend_service.training_checkpoint_files import atomic_json, read_json


def _failure(message: str) -> ApplicationFailure:
    """Keep budget failures readable without revealing storage details."""
    return ApplicationFailure("proof_budget_unavailable", message)


class ProofBudget:
    """Hold an exclusive proof lock and account for training, checks, and exports."""

    def __init__(
        self,
        root: Path,
        experiment_identifier: str,
        *,
        resume: bool = False,
        checkpoint_directory: Path | None = None,
    ) -> None:
        """Choose the shared project ledger and a stable experiment identity."""
        self.root = root
        self.experiment_identifier = experiment_identifier
        self.resume = resume
        self.checkpoint_directory = checkpoint_directory
        self.ledger = BudgetLedger()
        self._descriptor: int | None = None
        self._started: float | None = None
        self._allowance = 0.0
        self._finished_elapsed = 0.0

    @property
    def elapsed_seconds(self) -> float:
        """Measure this active invocation using a clock unaffected by wall changes."""
        if self._started is None:
            return self._finished_elapsed
        return max(0.0, time.monotonic() - self._started)

    @property
    def remaining_seconds(self) -> float:
        """Return the smaller remaining project and experiment wall-time allowance."""
        return max(0.0, self._allowance - self.elapsed_seconds)

    @property
    def training_remaining_seconds(self) -> float:
        """Reserve twenty minutes of the envelope for safe saves and proof checks."""
        return max(0.0, self.remaining_seconds - self.ledger.save_reserve_seconds)

    def check(self, *, training: bool = False) -> None:
        """Refuse another operation once its permitted time slice is exhausted."""
        remaining = (
            self.training_remaining_seconds if training else self.remaining_seconds
        )
        if self._descriptor is None or remaining <= 0:
            raise _failure(
                "The CPU proof time limit has been reached. Preserve the last "
                "checkpoint and report the measured result; do not start more work."
            )

    def _persist(self) -> None:
        """Validate and atomically publish the complete accounting record."""
        validated = BudgetLedger.model_validate(self.ledger.model_dump())
        atomic_json(self.root / "ledger.json", validated.model_dump(mode="json"))
        if not (self.root / ".initialized.json").exists():
            atomic_json(self.root / ".initialized.json", {"schema_version": 1})

    def _experiment(self, identifier: str) -> ExperimentBudget:
        """Find one already registered experiment without changing its allowance."""
        return next(
            item
            for item in self.ledger.experiments
            if item.experiment_identifier == identifier
        )

    def _reconcile(self) -> None:
        """Charge an abandoned operation's full reservation after an unclean exit."""
        active = self.ledger.active
        if active is None:
            return
        experiment = self._experiment(active.experiment_identifier)
        experiment.consumed_seconds += active.reserved_seconds
        self.ledger.consumed_seconds += active.reserved_seconds
        self.ledger.active = None
        self._persist()

    def _startup_retry(self) -> bool:
        """Allow a spent slot to retry only before any checkpoint was published."""
        directory = self.checkpoint_directory
        if directory is None:
            return False
        expected = self.root.parent.parent / "checkpoints" / self.experiment_identifier
        if directory.absolute() != expected.absolute():
            raise _failure("Use this experiment's original checkpoint directory.")
        for path in (directory.parent, directory):
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise _failure(
                    "Checkpoint storage is unsafe; restore it before retrying."
                )
        if not directory.exists():
            return True
        for path in directory.iterdir():
            if (
                path.is_symlink()
                or path.name != ".checkpoint.lock"
                or not path.is_file()
            ):
                return False
        return True

    def __enter__(self) -> "ProofBudget":
        """Lock, reconcile crashes, and reserve time before starting an operation."""
        try:
            ExperimentBudget(experiment_identifier=self.experiment_identifier)
            self.root.mkdir(parents=True, exist_ok=True)
            if self.root.is_symlink():
                raise ValueError("Ledger directories cannot be links.")
            self._descriptor = os.open(
                self.root / ".proof.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise _failure(
                    "Another CPU proof operation is already running."
                ) from None
            self._started = time.monotonic()
            ledger_path = self.root / "ledger.json"
            if ledger_path.exists() or ledger_path.is_symlink():
                self.ledger = BudgetLedger.model_validate_json(read_json(ledger_path))
            elif (self.root / ".initialized.json").exists():
                raise ValueError("The initialized project ledger is missing.")
            self._reconcile()
            existing = any(
                item.experiment_identifier == self.experiment_identifier
                for item in self.ledger.experiments
            )
            startup_retry = existing and not self.resume and self._startup_retry()
            if existing != self.resume and not startup_retry:
                raise _failure(
                    "Use resume for an existing proof experiment, or choose an unused "
                    "experiment identifier for a new experiment."
                )
            if not existing:
                if len(self.ledger.experiments) >= self.ledger.maximum_experiments:
                    raise _failure("Both CPU proof experiment slots have been used.")
                self.ledger.experiments.append(
                    ExperimentBudget(experiment_identifier=self.experiment_identifier)
                )
            experiment = self._experiment(self.experiment_identifier)
            self._allowance = min(
                self.ledger.maximum_total_seconds - self.ledger.consumed_seconds,
                self.ledger.maximum_experiment_seconds - experiment.consumed_seconds,
            )
            if self._allowance <= 0:
                raise _failure("The saved CPU proof time budget is exhausted.")
            self.ledger.active = ActiveBudget(
                experiment_identifier=self.experiment_identifier,
                started_at=time.time(),
                reserved_seconds=self._allowance,
            )
            self._persist()
            return self
        except ApplicationFailure:
            self._release()
            raise
        except Exception:
            self._release()
            raise _failure(
                "The CPU proof ledger could not be verified or saved. Check storage "
                "access and free space; keep the existing ledger for recovery."
            ) from None

    def _release(self) -> None:
        """Release ownership even when persistence or startup fails."""
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Charge actual elapsed time on a clean exit, including handled failures."""
        try:
            elapsed = self.elapsed_seconds
            experiment = self._experiment(self.experiment_identifier)
            experiment.consumed_seconds += elapsed
            self.ledger.consumed_seconds += elapsed
            self.ledger.active = None
            self._persist()
        except Exception:
            raise _failure(
                "The CPU proof accounting could not be saved. The reserved time "
                "will be charged during recovery; keep the existing ledger."
            ) from None
        finally:
            self._finished_elapsed = self.elapsed_seconds
            self._started = None
            self._release()
