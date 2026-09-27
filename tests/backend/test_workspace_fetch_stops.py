"""Fetch workers that go quiet in a network phase or die without a summary."""

from io import BytesIO
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fetch_job_fixtures import (
    EXECUTE,
    FakeExecution,
    Outcome,
    act,
    execute_next,
    fetch_request,
    make_service,
    open_partial,
    record,
)

from backend_service import workflow_execution
from backend_service.dataset_fetch_jobs import FETCH_INTERRUPTED, submit_fetch
from backend_service.failures import ApplicationFailure
from backend_service.workflow_execution import execution_failure
from schemas.base import StrictRecord
from schemas.dataset_fetch import DatasetFetchProgress

StopTrigger = str


@pytest.fixture(autouse=True)
def default_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the cache under the test workspace whatever the host configures."""
    monkeypatch.delenv("STEGOLAB_CACHE_DIRECTORY", raising=False)


def quiet_child(signals: list[int], exits_after: int | None) -> SimpleNamespace:
    """Build a child that never advances and ends when signalled or polled enough."""
    polls = count(1)

    def poll() -> int | None:
        """Report the stop exit code once signalled or once the poll budget is spent."""
        spent = exits_after is not None and next(polls) > exits_after
        return 143 if signals or spent else None

    return SimpleNamespace(
        stdin=BytesIO(),
        poll=poll,
        returncode=143,
        wait=lambda timeout=None: 143,
        send_signal=signals.append,
    )


def run_quiet_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, exits_after: int | None
) -> tuple[list[int], list[StrictRecord], ApplicationFailure]:
    """Run the executor against a worker that writes one record, then stays silent."""
    signals: list[int] = []

    def spawn(*arguments: object, **options: object) -> SimpleNamespace:
        """Write one progress record for the run label, then stay quiet."""
        assert cast(dict[str, str], options["env"])["STEGOLAB_DATA_DIRECTORY"] == "s"
        path = tmp_path / "logs" / "date_quiet_unique" / "fetch_progress.json"
        path.parent.mkdir(parents=True)
        path.write_text(record(1, phase=phase, bytes_received=5).model_dump_json())
        return quiet_child(signals, exits_after)

    ticks = count(0.0, 700.0)
    monkeypatch.setattr("backend_service.workflow_execution.subprocess.Popen", spawn)
    clock = SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
    monkeypatch.setattr(workflow_execution, "time", clock)
    progress: list[StrictRecord] = []
    with pytest.raises(ApplicationFailure) as raised:
        workflow_execution.execute_command(
            tmp_path,
            "fetch_dataset",
            {"run_label": "quiet"},
            on_process=lambda _: None,
            on_progress=progress.append,
            stop_requested=lambda: False,
            environment={"STEGOLAB_DATA_DIRECTORY": "s"},
        )
    return signals, progress, raised.value


def test_download_without_new_progress_is_stopped_as_stalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thirty minutes without a new sequence ends the worker with a fixed error."""
    signals, progress, failure = run_quiet_worker(
        tmp_path, monkeypatch, "downloading", None
    )
    assert failure.code == "workflow_stalled"
    assert "no progress for 30 minutes" in failure.message
    assert len(progress) == 1 and signals == [15]


def test_silent_preparation_is_not_a_stall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long inventory and integrity passes never touch the sink and may take hours."""
    signals, progress, failure = run_quiet_worker(tmp_path, monkeypatch, "preparing", 6)
    assert failure.code == "workflow_failed"
    assert len(progress) == 1 and signals == []


@pytest.mark.parametrize("trigger", ["cancel", "closing", "none"])
def test_worker_lost_without_summary_follows_the_requested_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trigger: StopTrigger
) -> None:
    """A stop that lands while resolving ends as asked; an unasked loss fails."""
    service = make_service(tmp_path)
    job = submit_fetch(service, fetch_request())
    partial = open_partial(service, job)

    def hook(_: DatasetFetchProgress) -> None:
        """Request the stop the scenario describes once progress is visible."""
        if trigger == "cancel":
            act(service, job, "cancel")
        elif trigger == "closing":
            service.closing.set()

    def lost(stopped: bool) -> Outcome:
        """Fail the way the executor does when the worker leaves no summary."""
        raise execution_failure("workflow_failed")

    fake = FakeExecution([record(1, phase="resolving")], lost, hook)
    monkeypatch.setattr(EXECUTE, fake)
    final = execute_next(service)
    expected = {"cancel": "cancelled", "closing": "interrupted", "none": "failed"}
    status = expected[trigger]
    assert (final.status, final.phase, final.available_actions) == (status, status, [])
    assert partial.exists() == (trigger != "cancel")
    if trigger == "none":
        assert final.error is not None and final.error["code"] == "workflow_failed"
    else:
        assert final.error == (FETCH_INTERRUPTED if trigger == "closing" else None)
