"""Storage fault handling without starting model or dataset workers."""

import threading
from pathlib import Path
from typing import cast

import pytest

from backend_service.failures import StorageFailure
from backend_service.workflow_supervision import run_scheduler
from backend_service.workspace_catalog import WorkspaceCatalog
from backend_service.workspace_jobs import WorkspaceJobService
from schemas.jobs import JobSnapshot
from schemas.workflows import WorkflowRequest


def queued_service(root: Path) -> tuple[WorkspaceJobService, JobSnapshot]:
    """Create one isolated accepted request while leaving the scheduler stopped."""
    source = root / "source"
    source.mkdir()
    catalog = WorkspaceCatalog(
        root, root / "application_state", source_roots={"Images": source}
    )
    service = WorkspaceJobService(catalog, root / "application_state")
    job = service.submit(
        WorkflowRequest(
            client_request_identifier="accepted",
            operation="prepare_dataset",
            source_identifier=catalog.snapshot().sources[0].identifier,
        )
    )
    return service, job


@pytest.mark.parametrize("failed_state", ["running", "completed"])
def test_failed_state_commit_halts_admission_and_hides_stale_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_state: str
) -> None:
    """Lost state commits halt admission and hide stale active snapshots."""
    service, job = queued_service(tmp_path)
    original = service.store.save
    launched: list[str] = []

    def save(snapshot: JobSnapshot, **options: object) -> JobSnapshot:
        """Inject disk exhaustion at one durable transition without changing files."""
        if snapshot.status == failed_state:
            raise StorageFailure()
        assert not options or options == {"mutation": None}
        return original(snapshot)

    def finish(supervisor: WorkspaceJobService, identifier: str) -> None:
        """Complete only simulated work to exercise a lost publication."""
        launched.append(identifier)
        supervisor._change(
            supervisor.store.get(identifier), status="completed", phase="completed"
        )

    monkeypatch.setattr(service.store, "save", save)
    monkeypatch.setattr("backend_service.workflow_supervision.run_job", finish)
    run_scheduler(service)
    assert service.persistence_failed
    assert service.closing.is_set()
    assert bool(launched) == (failed_state == "completed")
    with pytest.raises(StorageFailure):
        service.list_jobs()
    with pytest.raises(StorageFailure):
        service.get_job(job.job_identifier)
    with pytest.raises(StorageFailure):
        service.submit(cast(WorkflowRequest, service.store.request(job.job_identifier)))
    # The last committed record stays intact for startup reconciliation.
    assert service.store.get(job.job_identifier).status in ("queued", "running")
    service.close()
    recovered = WorkspaceJobService(service.catalog, tmp_path / "application_state")
    recovered.start()
    try:
        assert recovered.get_job(job.job_identifier).status == "interrupted"
        assert recovered.process is None
    finally:
        recovered.close()
    assert not (tmp_path / "state" / "cpu_proof").exists()


def test_shutdown_requests_stop_even_when_stop_intent_cannot_be_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken job database cannot prevent the independent shutdown signal."""
    service, job = queued_service(tmp_path)
    service._change(job, operation="train", status="running", phase="training")
    service.active_identifier = job.job_identifier
    observed_shutdown = threading.Event()

    def worker() -> None:
        """Observe the same independent stop flag used by the worker callback."""
        if service.closing.wait(timeout=2):
            observed_shutdown.set()

    def unavailable(*arguments: object, **options: object) -> JobSnapshot:
        """Reject the durable stop write without touching any experiment state."""
        raise StorageFailure()

    monkeypatch.setattr(service.store, "save", unavailable)
    service.thread = threading.Thread(target=worker)
    service.thread.start()
    service.close()
    assert observed_shutdown.is_set()
    assert service.persistence_failed
    assert not service.thread.is_alive()
    assert not (tmp_path / "state" / "cpu_proof").exists()
