"""Run one dataset fetch job in the worker and map its progress and exit."""

import logging
import os
import subprocess
from typing import TYPE_CHECKING

from backend_service.dataset_fetch_jobs import (
    FETCH_INTERRUPTED,
    STOPPING_ACTIONS,
    discard_partials,
    fetch_failure,
    resolve_fetch_document,
    run_label_for,
)
from backend_service.failures import ApplicationFailure
from backend_service.workflow_execution import execute_command, public_result
from schemas.base import StrictRecord
from schemas.dataset_fetch import DatasetFetchProgress, DatasetFetchRequest
from schemas.jobs import JobSnapshot

if TYPE_CHECKING:
    from backend_service.workspace_jobs import WorkspaceJobService

logger = logging.getLogger(__name__)

PHASE_BY_ACTION = {"cancel": "cancelling", "pause": "pausing"}
FINISHED_STATUSES = ("completed", "reused")
PAUSABLE_PHASE = "downloading"


def progress_ratio(record: DatasetFetchProgress) -> float | None:
    """Pick the counter that describes the current phase, or none at all.

    The worker carries the byte counters of the download forward into later
    records, so bytes are only meaningful while downloading; preparation is
    measured by its file counter and extraction has no honest counter.
    """
    if record.phase == "downloading" and record.bytes_total:
        return min(1.0, record.bytes_received / record.bytes_total)
    if record.phase == "preparing" and record.files_total:
        return min(1.0, record.files_completed / record.files_total)
    if record.phase == "completed":
        return 1.0
    return None


def progress_changes(
    job: JobSnapshot, record: DatasetFetchProgress
) -> dict[str, object]:
    """Translate one worker progress record into public snapshot fields."""
    metrics: dict[str, float | int] = {
        "sequence": record.sequence,
        "bytes_received": record.bytes_received,
        "files_completed": record.files_completed,
        "assets_completed": record.assets_completed,
        "assets_total": record.assets_total,
    }
    if record.bytes_total is not None:
        metrics["bytes_total"] = record.bytes_total
    if record.files_total is not None:
        metrics["files_total"] = record.files_total
    changes: dict[str, object] = {
        "phase": PHASE_BY_ACTION.get(job.requested_action or "", record.phase),
        "progress": progress_ratio(record),
        "metrics": metrics,
    }
    if job.requested_action is None:
        pausable = record.supports_pause and record.phase == PAUSABLE_PHASE
        changes["available_actions"] = ["cancel", "pause"] if pausable else ["cancel"]
    return changes


def stopped_changes(
    service: "WorkspaceJobService", job: JobSnapshot, partials: list[str]
) -> dict[str, object] | None:
    """Map a stop the runner asked for onto cancelled, paused or interrupted.

    Returns None when nobody asked the worker to stop, so the caller can
    treat the exit as a failure.
    """
    if job.requested_action == "cancel":
        discard_partials(
            service.catalog.cache_root, run_label_for(job.job_identifier), partials
        )
        return {"status": "cancelled", "phase": "cancelled", "available_actions": []}
    if job.requested_action == "pause":
        return {
            "status": "paused",
            "phase": "paused",
            "available_actions": ["resume", "cancel"],
        }
    if service.closing.is_set():
        return {
            "status": "interrupted",
            "phase": "interrupted",
            "available_actions": [],
            "error": dict(FETCH_INTERRUPTED),
        }
    return None


def exit_changes(
    service: "WorkspaceJobService",
    job: JobSnapshot,
    code: int,
    result: dict[str, object],
    partials: list[str],
) -> dict[str, object]:
    """Map the worker's exit code and summary onto the job's final state."""
    status = result.get("status")
    if code == 0 and status in FINISHED_STATUSES:
        return {
            "status": "completed",
            "phase": "completed",
            "progress": 1.0,
            "result": public_result(result),
            "available_actions": [],
            "error": None,
        }
    changes = stopped_changes(service, job, partials) if status == "stopped" else None
    if changes is None:
        raise fetch_failure("workflow_failed")
    return changes


def run_fetch_job(
    service: "WorkspaceJobService", identifier: str, request: DatasetFetchRequest
) -> None:
    """Run the fetch worker for one job, publishing progress and its outcome.

    A worker that dies without a summary after a stop was requested, which
    happens when the stop lands while it is still resolving the source, ends
    the job the way the request asked instead of as a failure.
    """
    document = resolve_fetch_document(service.catalog, identifier, request)
    partials: list[str] = []

    def remember_process(process: subprocess.Popen[bytes] | None) -> None:
        """Track process ownership only while this invocation is active."""
        service.process = process

    def stop_requested() -> bool:
        """Read persisted cancel or pause intent, including application shutdown."""
        return (
            service.closing.is_set()
            or service.store.get(identifier).requested_action in STOPPING_ACTIONS
        )

    def progress(record: StrictRecord) -> None:
        """Publish the worker's counters and remember its partial downloads."""
        if not isinstance(record, DatasetFetchProgress):
            return
        partials[:] = list(record.partial_directories)
        with service.lock:
            job = service.store.get(identifier)
            service._change(job, mutation=None, **progress_changes(job, record))

    try:
        code, result = execute_command(
            service.catalog.root,
            "fetch_dataset",
            document.model_dump(mode="json"),
            on_process=remember_process,
            on_progress=progress,
            stop_requested=stop_requested,
            environment={
                "STEGOLAB_DATA_DIRECTORY": os.path.abspath(service.store.path.parent)
            },
        )
    except ApplicationFailure as failure:
        if failure.code != "workflow_failed":
            raise
        with service.lock:
            job = service.store.get(identifier)
            changes = stopped_changes(service, job, partials)
            if changes is None:
                raise
            logger.info("fetch_worker_stopped_without_summary")
            service._change(job, mutation=None, **changes)
        return
    with service.lock:
        job = service.store.get(identifier)
        changes = exit_changes(service, job, code, result, partials)
        service._change(job, mutation=None, **changes)
