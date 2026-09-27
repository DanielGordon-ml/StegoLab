"""Admission, action rules and worker documents for dataset fetch jobs."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from backend_service.dataset_sources.cache import DatasetCache
from backend_service.dataset_sources.upload_sessions import upload_failure
from backend_service.failures import ApplicationFailure
from schemas.configuration import ConfigurationProfile
from schemas.dataset_fetch import DatasetFetchDocument, DatasetFetchRequest
from schemas.dataset_sources import UploadSourceSpec
from schemas.jobs import JobAction, JobSnapshot

if TYPE_CHECKING:
    from backend_service.workspace_catalog import WorkspaceCatalog
    from backend_service.workspace_jobs import WorkspaceJobService

logger = logging.getLogger(__name__)

FETCH_OPERATION = "fetch_dataset"
ACTIVE_FETCH_STATES = ("queued", "running", "paused")
STOPPING_ACTIONS = ("cancel", "pause")
PARTIAL_SUFFIX = ".partial"
FETCH_INTERRUPTED = {
    "code": "fetch_interrupted",
    "message": (
        "The application stopped while this download was running. "
        "Start it again to continue from the saved partial download."
    ),
}
_MESSAGES = {
    "dataset_fetch_busy": (
        "A dataset download is already queued, running or paused. "
        "Wait for it or cancel it first."
    ),
    "dataset_cleanup_busy": (
        "Unused downloads are being removed right now. Wait for the cleanup "
        "to finish, then try again."
    ),
    "dataset_uploads_unavailable": (
        "Dataset uploads are not available in this application."
    ),
    "action_unavailable": "This action is not supported for this job.",
    "service_stopping": "The application is stopping. Retry after restart.",
    "workflow_failed": (
        "The download could not finish. Check the source, free space and "
        "the run log, then start it again."
    ),
}
_STATUS_CODES = {
    "dataset_fetch_busy": 409,
    "dataset_cleanup_busy": 409,
    "action_unavailable": 409,
}


def fetch_failure(code: str) -> ApplicationFailure:
    """Return the fixed plain message for a fetch job failure code."""
    return ApplicationFailure(code, _MESSAGES[code], _STATUS_CODES.get(code, 503))


def is_fetch(job: JobSnapshot) -> bool:
    """Tell dataset fetch jobs apart from every other job kind."""
    return job.operation == FETCH_OPERATION


def run_label_for(job_identifier: str) -> str:
    """Derive the worker run label from a job identifier such as job_<hex>."""
    return job_identifier.removeprefix("job_")


def initial_running_actions(job: JobSnapshot) -> list[JobAction]:
    """Name the actions a job offers the moment the scheduler starts it."""
    if job.operation == "train":
        return ["stop"]
    if is_fetch(job):
        return ["cancel"]
    return []


def resolve_fetch_document(
    catalog: "WorkspaceCatalog", job_identifier: str, request: DatasetFetchRequest
) -> DatasetFetchDocument:
    """Build the private worker input with roots resolved by the catalog."""
    return DatasetFetchDocument(
        run_label=run_label_for(job_identifier),
        source=request.source,
        source_name=request.source_name,
        dataset_name=request.dataset_name,
        data_root=str(catalog.data_root),
        cache_root=str(catalog.cache_root),
        output_root=str(catalog.root / "datasets"),
        maximum_images=request.maximum_images,
        training_intended=request.training_intended,
        prepare=request.prepare,
        seed=0,
        resume=True,
    )


def discard_partials(
    cache_root: Path, run_label: str, reported: list[str] | None = None
) -> None:
    """Remove the partial downloads a job owns; leftovers are logged, not fatal."""
    cache = DatasetCache(cache_root)
    directories = set(reported or [])
    try:
        directories.update(cache.partial_directories(run_label))
    except (OSError, ApplicationFailure):
        logger.warning("fetch_partial_listing_failed")
    for directory in sorted(directories):
        kind, _, name = directory.partition("/")
        try:
            cache.remove_partial(
                kind, name.removesuffix(PARTIAL_SUFFIX), owner=run_label
            )
        except ApplicationFailure:
            logger.warning("fetch_partial_removal_failed")


def require_complete_upload(
    service: "WorkspaceJobService", upload_identifier: str
) -> None:
    """Refuse an upload source whose session is missing or still receiving parts."""
    if service.dataset_uploads is None:
        raise fetch_failure("dataset_uploads_unavailable")
    session = service.dataset_uploads.read(upload_identifier)
    if not session.complete:
        raise upload_failure(
            "upload_incomplete",
            missing=session.chunk_count - len(session.received_chunks),
            total=session.chunk_count,
        )


def submit_fetch(
    service: "WorkspaceJobService", request: DatasetFetchRequest
) -> JobSnapshot:
    """Accept one fetch at a time, replaying retries and refusing bad uploads."""
    service.require_storage()
    settings = request.model_dump(mode="json")
    fingerprint = service.store.fingerprint(settings)
    with service.lock:
        previous = service.store.replay(request.client_request_identifier, fingerprint)
        if previous is not None:
            return previous
        if service.closing.is_set():
            raise fetch_failure("service_stopping")
        if isinstance(request.source, UploadSourceSpec):
            require_complete_upload(service, request.source.upload_identifier)
        if service.cache_cleanup_in_progress:
            raise fetch_failure("dataset_cleanup_busy")
        if any(
            is_fetch(job) and job.status in ACTIVE_FETCH_STATES
            for job in service.store.list_jobs()
        ):
            raise fetch_failure("dataset_fetch_busy")
        now = datetime.now(UTC)
        snapshot = JobSnapshot(
            job_identifier="job_" + uuid4().hex,
            status="queued",
            phase="queued",
            configuration=ConfigurationProfile(),
            available_actions=["cancel"],
            created_at=now,
            updated_at=now,
            operation=FETCH_OPERATION,
            experiment_identifier=None,
            frozen_settings=settings,
        )
        accepted = service.store.save(
            snapshot,
            request=request,
            mutation=(request.client_request_identifier, fingerprint),
        )
        service.wake.set()
        return accepted


def fetch_action_changes(
    service: "WorkspaceJobService", job: JobSnapshot, action: JobAction
) -> dict[str, object]:
    """Describe the transition one action causes for a fetch job, or refuse it.

    A queued job may have run before it was paused and resumed, so cancelling
    it removes its partial downloads just like cancelling a paused job does.
    Resuming wakes the scheduler so the job does not wait for its next poll.
    """
    changes: dict[str, object] = {"requested_action": action, "available_actions": []}
    if action == "cancel" and job.status in ("queued", "paused"):
        discard_partials(service.catalog.cache_root, run_label_for(job.job_identifier))
        changes.update(status="cancelled", phase="cancelled")
    elif action == "cancel" and job.status == "running":
        changes["phase"] = "cancelling"
    elif (
        action == "pause"
        and job.status == "running"
        and "pause" in job.available_actions
    ):
        changes["phase"] = "pausing"
    elif action == "resume" and job.status == "paused":
        changes.update(
            status="queued",
            phase="queued",
            requested_action=None,
            available_actions=["cancel"],
        )
        service.wake.set()
    else:
        raise fetch_failure("action_unavailable")
    return changes
