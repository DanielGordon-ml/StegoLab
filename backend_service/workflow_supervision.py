"""Fail-closed queue execution and shutdown independent of metadata storage."""

import os
import subprocess
import time
from typing import TYPE_CHECKING
from uuid import uuid4

from backend_service.failures import ApplicationFailure, StorageFailure
from backend_service.inference_jobs import is_inference
from backend_service.workflow_lock import proof_busy
from backend_service.workflow_runner import run_job
from schemas.jobs import JobSnapshot

if TYPE_CHECKING:
    from backend_service.workspace_jobs import WorkspaceJobService

MAINTENANCE_INTERVAL_SECONDS = 60.0


def next_job(service: "WorkspaceJobService") -> JobSnapshot | None:
    """Persist ownership before launching work; failed writes launch nothing."""
    with service.lock:
        proof_locked = proof_busy(service.catalog.root)
        queued = next(
            (
                job
                for job in service.store.list_jobs()
                if job.status == "queued"
                and not (
                    proof_locked and job.operation in ("train", "evaluate", "export")
                )
            ),
            None,
        )
        if queued is not None:
            service._change(
                queued,
                status="running",
                phase="preparing",
                available_actions=["stop"] if queued.operation == "train" else [],
            )
            service.active_identifier = queued.job_identifier
        return queued


def report_failure(
    service: "WorkspaceJobService", identifier: str, failure: Exception
) -> None:
    """Record safe operation errors; let storage failures halt the entire queue."""
    reference = uuid4().hex
    code, message = (
        "workflow_failed",
        (
            "The operation could not finish. "
            "Check saved data, available time, and storage."
        ),
    )
    if isinstance(failure, ApplicationFailure):
        code, message = failure.code, failure.message
    if service.logger is not None:
        service.logger.error(
            "workflow_failed", extra={"diagnostic_reference": reference}
        )
    with service.lock:
        job = service.store.get(identifier)
        if service.closing.is_set() and is_inference(job):
            service.interrupt(job)
            return
        state = "interrupted" if service.closing.is_set() else "failed"
        service._change(
            job,
            status=state,
            phase=state,
            available_actions=[],
            error={"code": code, "message": message, "diagnostic_reference": reference},
        )


def run_scheduler(service: "WorkspaceJobService") -> None:
    """Halt admission when durable state fails instead of stranding accepted jobs."""
    maintained = time.monotonic()
    try:
        while not service.closing.is_set():
            queued = next_job(service)
            if queued is None:
                if time.monotonic() - maintained >= MAINTENANCE_INTERVAL_SECONDS:
                    maintained = time.monotonic()
                    maintain(service)
                service.wake.wait(timeout=0.5)
                service.wake.clear()
                continue
            try:
                run_job(service, queued.job_identifier)
            except StorageFailure:
                raise
            except Exception as failure:
                report_failure(service, queued.job_identifier, failure)
            finally:
                with service.lock:
                    service.active_identifier = None
    except Exception:
        service.halt_for_storage()


def maintain(service: "WorkspaceJobService") -> None:
    """Forget expired inference files and texts without stopping the queue."""
    if service.inference is None:
        return
    try:
        service.inference.maintain()
    except ApplicationFailure:
        if service.logger is not None:
            service.logger.error("inference_maintenance_failed")


def close_supervisor(service: "WorkspaceJobService") -> None:
    """Prioritize safe worker shutdown even if the job database cannot be written."""
    service.closing.set()
    try:
        with service.lock:
            if service.active_identifier is not None:
                job = service.store.get(service.active_identifier)
                if job.operation == "train" and job.status == "running":
                    service._change(
                        job,
                        requested_action="stop",
                        phase="saving",
                        available_actions=[],
                    )
    except Exception:
        service.halt_for_storage()
    service.wake.set()
    if service.thread is not None:
        service.thread.join(timeout=30)
        process = service.process
        if service.thread.is_alive() and process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            service.thread.join(timeout=2)
    try:
        # Only the supervisor that owns the queue lock may reconcile the queue;
        # a refused second instance must leave the owner's jobs untouched.
        if not service.persistence_failed and service.ownership is not None:
            with service.lock:
                for job in service.store.list_jobs():
                    if job.status in ("queued", "running"):
                        service.interrupt(job)
    except Exception:
        service.halt_for_storage()
    finally:
        if (
            service.thread is None or not service.thread.is_alive()
        ) and service.ownership is not None:
            os.close(service.ownership)
            service.ownership = None
