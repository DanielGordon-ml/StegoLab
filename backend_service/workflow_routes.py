"""Workflow submission, safe actions, preflight, and reconnectable events."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend_service.failures import ApplicationFailure
from backend_service.workspace_jobs import WorkspaceJobService
from schemas.errors import ErrorEnvelope
from schemas.jobs import JobSnapshot
from schemas.workflows import TrainingPreflight, WorkflowActionRequest, WorkflowRequest

job_router = APIRouter(
    prefix="/api/v1",
    responses={
        status: {"model": ErrorEnvelope} for status in (404, 409, 422, 500, 503)
    },
)


def service(request: Request) -> WorkspaceJobService:
    """Read the one supervisor created during application startup."""
    return cast(WorkspaceJobService, request.app.state.workspace_jobs)


def submit(payload: WorkflowRequest, request: Request, operation: str) -> JobSnapshot:
    """Reject a mismatched endpoint before any work or retry registration."""
    if payload.operation != operation:
        raise ApplicationFailure(
            "operation_mismatch", "Choose the matching operation for this request.", 422
        )
    return service(request).submit(payload)


@job_router.post("/training_preflight", response_model=TrainingPreflight)
def preflight(payload: WorkflowRequest, request: Request) -> TrainingPreflight:
    """Inspect a request without writing experiment state or spending time."""
    return service(request).preflight(payload)


@job_router.post("/training_jobs", response_model=JobSnapshot, status_code=202)
def training(payload: WorkflowRequest, request: Request) -> JobSnapshot:
    """Queue one fixed CPU training attempt or compatible checkpoint continuation."""
    return submit(payload, request, "train")


@job_router.post("/evaluation_jobs", response_model=JobSnapshot, status_code=202)
def evaluation(payload: WorkflowRequest, request: Request) -> JobSnapshot:
    """Queue measured checks for a registered checkpoint and dataset."""
    return submit(payload, request, "evaluate")


@job_router.post("/export_jobs", response_model=JobSnapshot, status_code=202)
def export(payload: WorkflowRequest, request: Request) -> JobSnapshot:
    """Queue independent experimental packages without enabling public inference."""
    return submit(payload, request, "export")


@job_router.post("/dataset_jobs", response_model=JobSnapshot, status_code=202)
def dataset(payload: WorkflowRequest, request: Request) -> JobSnapshot:
    """Prepare a registered local folder using the immutable dataset service."""
    return submit(payload, request, "prepare_dataset")


@job_router.post("/jobs/{job_identifier}/actions", response_model=JobSnapshot)
def action(
    job_identifier: str, payload: WorkflowActionRequest, request: Request
) -> JobSnapshot:
    """Persist a supported action once even when acknowledgement is lost."""
    return service(request).action(job_identifier, payload)


async def event_stream(
    request: Request, supervisor: WorkspaceJobService
) -> AsyncGenerator[str, None]:
    """Replay durable events or send an atomic snapshot when the cursor is absent."""
    raw = request.headers.get("last-event-id") or request.query_params.get("after")
    try:
        cursor = int(raw) if raw is not None else -1
    except ValueError:
        cursor = -1
    if cursor < 0 or cursor > supervisor.store.latest_event():
        with supervisor.lock:
            cursor = supervisor.store.latest_event()
            snapshots = {job.job_identifier: job for job in supervisor.list_jobs()}
            original_store = getattr(request.app.state, "store", None)
            if original_store is not None:
                for job in original_store.list_jobs():
                    snapshots.setdefault(job.job_identifier, job)
            payload = {
                "items": [job.model_dump(mode="json") for job in snapshots.values()]
            }
        yield f"id: {cursor}\nevent: reset\ndata: {json.dumps(payload)}\n\n"
    heartbeat = time.monotonic()
    opened = heartbeat
    while not await request.is_disconnected():
        supervisor.require_storage()
        # Bound connection lifetime so restart can drain open browser streams.
        if time.monotonic() - opened >= 10:
            return
        for event in supervisor.store.events(cursor):
            cursor = event.event_identifier
            yield f"id: {cursor}\nevent: job\ndata: {event.model_dump_json()}\n\n"
        if time.monotonic() - heartbeat >= 3:
            yield ": heartbeat\n\n"
            heartbeat = time.monotonic()
        await asyncio.sleep(0.25)


@job_router.get("/events")
def events(request: Request) -> StreamingResponse:
    """Keep one same-origin event stream with replay and no intermediary buffering."""
    return StreamingResponse(
        event_stream(request, service(request)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
