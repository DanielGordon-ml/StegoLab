"""Durable browser workflows without spending the real research allowance."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import FastAPI, Request

from backend_service.failures import ApplicationFailure
from backend_service.workflow_preflight import read_budget
from backend_service.workflow_routes import event_stream
from backend_service.workspace_catalog import WorkspaceCatalog
from backend_service.workspace_jobs import WorkspaceJobService
from schemas.workflows import WorkflowActionRequest, WorkflowRequest


def make_service(root: Path) -> tuple[WorkspaceJobService, WorkflowRequest]:
    """Register an isolated folder while leaving actual command execution stopped."""
    root.mkdir(exist_ok=True)
    source = root / "source"
    source.mkdir(exist_ok=True)
    catalog = WorkspaceCatalog(
        root, root / "application_state", source_roots={"Images": source}
    )
    identifier = catalog.snapshot().sources[0].identifier
    service = WorkspaceJobService(catalog, root / "application_state")
    return service, WorkflowRequest(
        client_request_identifier="prepare_once",
        operation="prepare_dataset",
        source_identifier=identifier,
    )


def test_submission_and_action_replay_survive_reload(tmp_path: Path) -> None:
    """A lost browser acknowledgement cannot duplicate a job or repeat cancellation."""
    service, request = make_service(tmp_path)
    first = service.submit(request)
    assert service.submit(request).job_identifier == first.job_identifier
    reloaded, _ = make_service(tmp_path)
    assert reloaded.submit(request).job_identifier == first.job_identifier
    action = WorkflowActionRequest(
        client_request_identifier="cancel_once", action="cancel"
    )
    cancelled = reloaded.action(first.job_identifier, action)
    assert cancelled.status == "cancelled"
    assert service.action(first.job_identifier, action).status == "cancelled"
    assert len(service.list_jobs()) == 1
    assert len(service.store.events(0)) == 2
    changed = request.model_copy(update={"dataset_name": "other_images"})
    with pytest.raises(ApplicationFailure, match="another change"):
        service.submit(changed)
    with pytest.raises(ApplicationFailure, match="no longer available"):
        service.action(
            first.job_identifier,
            WorkflowActionRequest(
                client_request_identifier="stop_later", action="stop"
            ),
        )
    assert not (tmp_path / "state" / "cpu_proof").exists()


def test_restart_marks_unfinished_jobs_interrupted(tmp_path: Path) -> None:
    """Restart never starts queued training or preparation automatically."""
    service, request = make_service(tmp_path)
    job = service.submit(request)
    reloaded, _ = make_service(tmp_path)
    reloaded.start()
    try:
        snapshot = reloaded.get_job(job.job_identifier)
        assert snapshot.status == "interrupted"
        assert snapshot.available_actions == []
        assert reloaded.process is None
        with pytest.raises(ApplicationFailure, match="Another application"):
            service.start()
    finally:
        reloaded.close()
    assert not (tmp_path / "datasets").exists()


def test_sse_replays_ordered_events_and_resets_unknown_cursor(tmp_path: Path) -> None:
    """Replay complete events and reset invalid positions to current history."""
    service, request = make_service(tmp_path)
    job = service.submit(request)
    service.action(
        job.job_identifier,
        WorkflowActionRequest(client_request_identifier="cancel", action="cancel"),
    )

    async def first_message(cursor: str) -> str:
        """Read one event directly without a long-lived browser connection."""
        application = FastAPI()
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/events",
            "query_string": b"",
            "headers": [(b"last-event-id", cursor.encode())],
            "app": application,
        }

        async def receive() -> dict[str, object]:
            """Keep the synthetic request connected for its first response."""
            return {"type": "http.request", "body": b"", "more_body": False}

        stream = event_stream(Request(scope, receive=receive), service)
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    replay = asyncio.run(first_message("1"))
    assert "id: 2\nevent: job" in replay
    event = json.loads(replay.split("data: ", 1)[1])
    assert event["snapshot"]["status"] == "cancelled"
    assert event["snapshot"]["latest_event_identifier"] == 2
    reset = asyncio.run(first_message("9999"))
    assert "event: reset" in reset
    payload = json.loads(reset.split("data: ", 1)[1])
    assert payload["items"][0]["job_identifier"] == job.job_identifier


def test_missing_ledger_never_restarts_used_allowance(tmp_path: Path) -> None:
    """Existing checkpoints block new allowances without initialization markers."""
    index = tmp_path / "checkpoints" / "original" / "index.json"
    index.parent.mkdir(parents=True)
    index.write_text("{}")
    with pytest.raises(ValueError, match="ledger is missing"):
        read_budget(tmp_path)
    assert not (tmp_path / "state").exists()


def test_preflight_is_read_only_and_reports_budget_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preview checks capacity without creating or spending the ledger."""
    service, _ = make_service(tmp_path)
    monkeypatch.setattr(service.catalog, "dataset_blockers", lambda _: [])
    request = WorkflowRequest(
        client_request_identifier="preview",
        operation="train",
        dataset_identifier="known_sample",
        cpu_threads=2,
    )
    before = set(tmp_path.rglob("*"))
    preview = service.preflight(request)
    assert preview.allowed
    assert preview.resolved_settings["cpu_threads"] == 2
    assert preview.resolved_settings["checkpoint_interval_seconds"] == 300
    assert set(tmp_path.rglob("*")) == before
    directory = tmp_path / "state" / "cpu_proof"
    directory.mkdir(parents=True)
    directory.joinpath("ledger.json").write_text(
        json.dumps(
            {
                "consumed_seconds": 14400.0,
                "experiments": [
                    {"experiment_identifier": "one", "consumed_seconds": 7200.0},
                    {"experiment_identifier": "two", "consumed_seconds": 7200.0},
                ],
            }
        )
    )
    exhausted = service.preflight(request)
    assert not exhausted.allowed
    assert exhausted.remaining_budget_seconds == 0
    assert exhausted.remaining_experiment_slots == 0
