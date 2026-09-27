"""Shared queue, request, progress and fake worker helpers for fetch job tests."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import pytest

from backend_service.dataset_fetch_jobs import submit_fetch
from backend_service.dataset_sources.cache import DatasetCache
from backend_service.dataset_sources.plans import PlannedAsset
from backend_service.failures import ApplicationFailure
from backend_service.workflow_runner import run_job
from backend_service.workflow_supervision import next_job, report_failure
from backend_service.workspace_catalog import WorkspaceCatalog
from backend_service.workspace_jobs import WorkspaceJobService
from schemas.base import StrictRecord
from schemas.dataset_fetch import DatasetFetchProgress, DatasetFetchRequest
from schemas.dataset_sources import HttpsArchiveSourceSpec
from schemas.jobs import JobSnapshot
from schemas.workflows import WorkflowActionRequest

IDENTITY = "c" * 64
KIND = "https_archive"
URL = "https://example.test/archive.zip"
EXECUTE = "backend_service.workflow_fetch_runner.execute_command"
STOPPED: dict[str, object] = {"status": "stopped", "stop_signal": 15}
COMPLETED: dict[str, object] = {"status": "completed", "member_count": 2}
Outcome = tuple[int, dict[str, object]]
ActionName = Literal["stop", "cancel", "pause", "resume"]


def fetch_request(identifier: str = "fetch_once") -> DatasetFetchRequest:
    """Describe one archive fetch with no secrets."""
    return DatasetFetchRequest(
        client_request_identifier=identifier,
        source=HttpsArchiveSourceSpec(url=URL, terms_reference="not_reviewed"),
        source_name="archive",
        dataset_name="archive_prepared",
    )


def make_service(root: Path) -> WorkspaceJobService:
    """Build a queue over an isolated workspace without starting its scheduler."""
    catalog = WorkspaceCatalog(root, root / "application_state", source_roots={})
    return WorkspaceJobService(catalog, root / "application_state")


def record(sequence: int, **fields: object) -> DatasetFetchProgress:
    """Build a worker progress record with the given counters."""
    stamp = "2026-09-27T00:00:00+00:00"
    values = {"sequence": sequence, "phase": "downloading", "updated_at": stamp}
    return DatasetFetchProgress.model_validate({**values, **fields})


class FakeExecution:
    """Replace the worker: publish scripted progress, then report an exit."""

    def __init__(
        self,
        records: list[DatasetFetchProgress],
        outcome: Callable[[bool], Outcome] = lambda stopped: (
            (143, STOPPED) if stopped else (0, COMPLETED)
        ),
        after_record: Callable[[DatasetFetchProgress], None] = lambda _: None,
    ) -> None:
        """Remember the script and the hook that runs after each record."""
        self.records, self.outcome, self.after_record = records, outcome, after_record
        self.calls: list[tuple[dict[str, object], object]] = []

    def __call__(self, *arguments: object, **options: object) -> Outcome:
        """Publish every scripted record, then answer with the scripted exit."""
        document = cast(dict[str, object], arguments[2])
        self.calls.append((document, options.get("environment")))
        on_progress = cast(Callable[[StrictRecord], None], options["on_progress"])
        for item in self.records:
            on_progress(item)
            self.after_record(item)
        return self.outcome(cast(Callable[[], bool], options["stop_requested"])())


def execute_next(service: WorkspaceJobService) -> JobSnapshot:
    """Run one scheduler turn synchronously, recording failures like the loop."""
    queued = next_job(service)
    assert queued is not None
    try:
        run_job(service, queued.job_identifier)
    except ApplicationFailure as failure:
        report_failure(service, queued.job_identifier, failure)
    service.active_identifier = None
    return service.get_job(queued.job_identifier)


def act(
    service: WorkspaceJobService, job: JobSnapshot, name: ActionName
) -> JobSnapshot:
    """Send one browser action to a job, retried under the action's own name."""
    request = WorkflowActionRequest(client_request_identifier=name, action=name)
    return service.action(job.job_identifier, request)


def open_partial(service: WorkspaceJobService, job: JobSnapshot) -> Path:
    """Leave a real partial download in the cache on behalf of one job."""
    asset = PlannedAsset(URL, "archive.zip", 200, None, None, True, None)
    cache = DatasetCache(service.catalog.cache_root)
    owner = job.job_identifier.removeprefix("job_")
    return cache.open_partial(KIND, IDENTITY, asset, owner=owner).directory


def pause_job(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[WorkspaceJobService, JobSnapshot, FakeExecution]:
    """Run one fetch until the browser pauses it at its first progress record."""
    service = make_service(root)
    job = submit_fetch(service, fetch_request())
    seen: list[JobSnapshot] = []
    fake = FakeExecution(
        [record(1, bytes_received=20, bytes_total=200, supports_pause=True)],
        after_record=lambda _: seen.append(act(service, job, "pause")),
    )
    monkeypatch.setattr(EXECUTE, fake)
    paused = execute_next(service)
    assert (seen[0].phase, seen[0].available_actions) == ("pausing", [])
    assert (paused.status, paused.phase) == ("paused", "paused")
    assert paused.available_actions == ["resume", "cancel"]
    assert paused.progress == 0.1 and paused.metrics["bytes_received"] == 20
    return service, paused, fake
