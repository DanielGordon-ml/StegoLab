"""Small atomic ledger transactions independent of the model-operation lock."""

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from backend_service.failures import ApplicationFailure
from backend_service.training_checkpoint_files import atomic_json, read_json
from schemas.pilot_budget import PilotBudgetLedger

RESTORE_MESSAGE = (
    "The pilot ledger is unavailable or invalid. Restore its verified backup; do "
    "not create a replacement budget."
)


def budget_failure(message: str) -> ApplicationFailure:
    """Return public accounting guidance without file contents or private paths."""
    return ApplicationFailure("pilot_budget_unavailable", message)


def _acquire(directory: Path, initialize: bool) -> int:
    """Open and lock the ledger directory, sharing the lock with a root host guard."""
    if initialize:
        directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("The ledger directory is unavailable.")
    descriptor = os.open(
        directory / ".ledger.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o660
    )
    try:
        if os.geteuid() == 0:
            owner = directory.stat()
            os.fchown(descriptor, owner.st_uid, owner.st_gid)
        if os.fstat(descriptor).st_uid == os.geteuid():
            os.fchmod(descriptor, 0o660)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def ledger_lock(directory: Path, *, initialize: bool = False) -> Iterator[None]:
    """Serialize short state operations while letting the host guard read safely.

    Only failures of the lock itself become storage failures; what happens inside
    the lock reports its own reason, so an operator mistake never reads as a
    damaged ledger.
    """
    try:
        descriptor = _acquire(directory, initialize)
    except BlockingIOError:
        raise ApplicationFailure(
            "pilot_budget_busy", "Another ledger update is running. Retry shortly."
        ) from None
    except (OSError, ValueError):
        raise budget_failure(RESTORE_MESSAGE) from None
    try:
        yield
    finally:
        os.close(descriptor)


def load_ledger(directory: Path) -> PilotBudgetLedger:
    """Fail closed on missing or damaged accounting; never initialize on read."""
    try:
        marker = json.loads(read_json(directory / ".initialized.json"))
        if marker != {"schema_version": 1} or type(marker["schema_version"]) is not int:
            raise ValueError("The initialization record is invalid.")
        return PilotBudgetLedger.model_validate_json(
            read_json(directory / "ledger.json")
        )
    except (OSError, ValueError):
        raise budget_failure(RESTORE_MESSAGE) from None


def save_ledger(directory: Path, ledger: PilotBudgetLedger) -> None:
    """Verify records before publishing them atomically with directory flushing."""
    try:
        validated = PilotBudgetLedger.model_validate_json(ledger.model_dump_json())
    except ValidationError as failure:
        reason = failure.errors()[0]["msg"].removeprefix("Value error, ")
        raise ApplicationFailure(
            "pilot_budget_inconsistent",
            f"The change would break the ledger rules: {reason} Nothing was saved.",
            422,
        ) from None
    try:
        atomic_json(directory / "ledger.json", validated.model_dump(mode="json"))
    except (OSError, ValueError):
        raise budget_failure(RESTORE_MESSAGE) from None


def initialize_pilot_budget(directory: Path) -> PilotBudgetLedger:
    """Explicitly initialize empty operator-owned storage before any EC2 launch."""
    with ledger_lock(directory, initialize=True):
        if any(path.name != ".ledger.lock" for path in directory.iterdir()):
            raise budget_failure("Pilot ledger storage is not empty; keep its records.")
        ledger = PilotBudgetLedger()
        save_ledger(directory, ledger)
        try:
            atomic_json(directory / ".initialized.json", {"schema_version": 1})
        except (OSError, ValueError):
            raise budget_failure(RESTORE_MESSAGE) from None
        return ledger
