"""Fetch jobs on the queue: progress, cancel, pause, resume and shutdown."""

from pathlib import Path

import pytest
from fetch_job_fixtures import (
    COMPLETED,
    EXECUTE,
    IDENTITY,
    KIND,
    STOPPED,
    FakeExecution,
    Outcome,
    act,
    execute_next,
    fetch_request,
    make_service,
    open_partial,
    pause_job,
    record,
)

from backend_service.dataset_fetch_jobs import FETCH_INTERRUPTED, submit_fetch
from backend_service.failures import ApplicationFailure
from schemas.jobs import JobSnapshot


@pytest.fixture(autouse=True)
def default_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the cache under the test workspace whatever the host configures."""
    monkeypatch.delenv("STEGOLAB_CACHE_DIRECTORY", raising=False)


def test_progress_maps_to_phase_progress_metrics_and_pause_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counters become public fields; only documented actions are ever accepted."""
    service = make_service(tmp_path)
    job = submit_fetch(service, fetch_request())
    assert (job.status, job.phase) == ("queued", "queued")
    assert (job.available_actions, job.operation) == (["cancel"], "fetch_dataset")
    assert submit_fetch(service, fetch_request()).job_identifier == job.job_identifier
    with pytest.raises(ApplicationFailure, match="already queued"):
        submit_fetch(service, fetch_request("second"))
    with pytest.raises(ApplicationFailure, match="no longer available"):
        act(service, job, "pause")
    seen: list[JobSnapshot] = []
    downloaded = {"bytes_received": 200, "bytes_total": 200, "supports_pause": True}
    counted = {"files_completed": 1, "files_total": 4, **downloaded}
    fake = FakeExecution(
        [
            record(1, phase="resolving"),
            record(2, bytes_received=50, bytes_total=200, supports_pause=True),
            record(3, phase="extracting", **downloaded),
            record(4, phase="preparing", **counted),
        ],
        after_record=lambda _: seen.append(service.get_job(job.job_identifier)),
    )
    monkeypatch.setattr(EXECUTE, fake)
    final = execute_next(service)
    assert (seen[0].phase, seen[0].progress) == ("resolving", None)
    assert seen[0].available_actions == ["cancel"]
    assert (seen[1].phase, seen[1].progress) == ("downloading", 0.25)
    assert seen[1].available_actions == ["cancel", "pause"]
    assert (seen[1].metrics["sequence"], seen[1].metrics["bytes_total"]) == (2, 200)
    assert "files_total" not in seen[1].metrics and seen[1].metrics["assets_total"] == 0
    # Byte counters carried forward from the download no longer drive the bar,
    # and pause is only offered while bytes are still arriving.
    assert (seen[2].phase, seen[2].progress) == ("extracting", None)
    assert seen[2].available_actions == ["cancel"]
    assert (seen[3].progress, seen[3].metrics["files_total"]) == (0.25, 4)
    assert seen[3].available_actions == ["cancel"]
    assert (final.status, final.phase, final.progress) == ("completed", "completed", 1)
    assert final.result == COMPLETED and final.available_actions == []
    document, environment = fake.calls[0]
    assert document["run_label"] == job.job_identifier.removeprefix("job_")
    assert document["resume"] is True and document["seed"] == 0
    assert document["data_root"] == str(tmp_path / "data")
    assert document["cache_root"] == str(tmp_path / ".cache/stegolab/datasets")
    assert environment == {"STEGOLAB_DATA_DIRECTORY": str(service.store.path.parent)}
    running = service._change(final, status="running", available_actions=["stop"])
    with pytest.raises(ApplicationFailure, match="not supported") as unsupported:
        act(service, running, "stop")
    assert unsupported.value.status_code == 409
    queued = service._change(running, status="queued", available_actions=["cancel"])
    assert act(service, queued, "cancel").status == "cancelled"
    assert submit_fetch(service, fetch_request("after_cancel")).status == "queued"


def test_worker_receives_the_absolute_state_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative state directory reaches the worker made absolute by the API."""
    monkeypatch.chdir(tmp_path)
    service = make_service(Path("workspace"))
    submit_fetch(service, fetch_request())
    fake = FakeExecution([record(1, phase="resolving")])
    monkeypatch.setattr(EXECUTE, fake)
    assert execute_next(service).status == "completed"
    expected = str(Path.cwd() / "workspace" / "application_state")
    assert fake.calls[0][1] == {"STEGOLAB_DATA_DIRECTORY": expected}


def test_cancel_while_running_removes_the_partial_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A running cancel is acknowledged at once and clears what the job owned."""
    service = make_service(tmp_path)
    job = submit_fetch(service, fetch_request())
    partial = open_partial(service, job)
    seen: list[JobSnapshot] = []
    reported = [f"{KIND}/{IDENTITY}.partial"]
    fake = FakeExecution(
        [
            record(1, bytes_received=10, bytes_total=200, supports_pause=True),
            record(2, bytes_received=30, bytes_total=200, partial_directories=reported),
        ],
        after_record=lambda item: seen.append(
            act(service, job, "cancel")
            if item.sequence == 1
            else service.get_job(job.job_identifier)
        ),
    )
    monkeypatch.setattr(EXECUTE, fake)
    final = execute_next(service)
    assert (seen[0].phase, seen[0].requested_action) == ("cancelling", "cancel")
    assert (seen[1].phase, seen[1].available_actions) == ("cancelling", [])
    assert (final.status, final.phase, final.result) == ("cancelled", "cancelled", None)
    assert not partial.exists()


def test_paused_jobs_resume_the_same_request_or_cancel_their_partials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume wakes the scheduler and reruns the request; cancel frees the cache."""
    service, paused, first = pause_job(tmp_path / "resume", monkeypatch)
    service.wake.clear()
    resumed = act(service, paused, "resume")
    assert (resumed.status, resumed.phase) == ("queued", "queued")
    assert resumed.available_actions == ["cancel"] and resumed.requested_action is None
    assert service.wake.is_set()
    second = FakeExecution([record(1, bytes_received=200, bytes_total=200)])
    monkeypatch.setattr(EXECUTE, second)
    final = execute_next(service)
    assert (final.status, final.progress) == ("completed", 1)
    assert second.calls[0][0]["run_label"] == first.calls[0][0]["run_label"]
    other, paused, _ = pause_job(tmp_path / "cancel", monkeypatch)
    partial = open_partial(other, paused)
    cancelled = act(other, paused, "cancel")
    assert (cancelled.status, cancelled.phase) == ("cancelled", "cancelled")
    assert cancelled.available_actions == [] and not partial.exists()
    # A job cancelled while queued after a resume still owns partial downloads.
    third, paused, _ = pause_job(tmp_path / "queued_cancel", monkeypatch)
    partial = open_partial(third, paused)
    queued = act(third, paused, "resume")
    cancelled = act(third, queued, "cancel")
    assert (cancelled.status, partial.exists()) == ("cancelled", False)


@pytest.mark.parametrize("closing", [False, True])
@pytest.mark.parametrize("outcome", [(1, {}), (143, STOPPED), (0, STOPPED)])
def test_unrequested_stops_fail_unless_the_application_is_closing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: Outcome, closing: bool
) -> None:
    """Failures and unrequested stops fail; a shutdown interrupts with a hint."""
    service = make_service(tmp_path)
    submit_fetch(service, fetch_request())
    hook = (lambda _: service.closing.set()) if closing else (lambda _: None)
    records = [record(1, bytes_received=10, bytes_total=200)]
    monkeypatch.setattr(EXECUTE, FakeExecution(records, lambda _: outcome, hook))
    final = execute_next(service)
    status = "interrupted" if closing else "failed"
    assert (final.status, final.phase, final.available_actions) == (status, status, [])
    hinted = closing and outcome[1].get("status") == "stopped"
    assert final.error is not None and final.error["code"] == (
        "fetch_interrupted" if hinted else "workflow_failed"
    )
    assert not hinted or final.error == FETCH_INTERRUPTED
