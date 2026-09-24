"""Bound local preparation separately from explicitly opened GPU host sessions."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from backend_service.failures import ApplicationFailure
from schemas.pilot_training import PilotRequest


@contextmanager
def pilot_execution(request: PilotRequest) -> Iterator[float]:
    """Yield the work deadline; GPU operation ownership lasts through saving."""
    if request.execution_mode == "cpu_smoke":
        reserve = min(10.0, request.cpu_smoke_timeout_seconds / 10.0)
        yield time.monotonic() + request.cpu_smoke_timeout_seconds - reserve
        return
    if request.session_identifier is None:
        raise ApplicationFailure("pilot_session", "Open an explicit GPU session first.")
    from backend_service.pilot_budget import PilotOperation

    directory = Path(request.output_root).resolve() / "state" / "gpu_pilot"
    with PilotOperation(directory, request.session_identifier) as deadline:
        yield deadline.checkpoint_monotonic_deadline
