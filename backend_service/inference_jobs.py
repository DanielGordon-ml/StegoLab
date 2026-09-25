"""Admission, secret hand-off, and lifecycle helpers for encode and decode jobs."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from backend_service.failures import ApplicationFailure
from backend_service.inference_admission import admit_inference
from backend_service.inference_files import InferenceFileStore
from backend_service.inference_secrets import (
    DecodedTextStore,
    InferenceSecretStore,
    queue_full_failure,
)
from backend_service.model_installation import InstalledModelStore
from schemas.configuration import ConfigurationProfile
from schemas.inference_jobs import (
    MAXIMUM_INFERENCE_JOBS,
    DecodingJobRequest,
    EncodingJobRequest,
)
from schemas.jobs import JobSnapshot

if TYPE_CHECKING:
    from backend_service.workspace_jobs import WorkspaceJobService

INPUTS_LOST = {
    "code": "inputs_lost",
    "message": "The application restarted before this job finished. The message "
    "and password were not saved. Start the job again.",
}


@dataclass
class InferenceServices:
    """Stores an inference job needs, attached to the one job supervisor."""

    files: InferenceFileStore
    installed: InstalledModelStore
    secrets: InferenceSecretStore
    texts: DecodedTextStore

    def maintain(self) -> None:
        """Forget expired uploads, results, scratch folders, and texts."""
        with self.files.lock:
            self.files.sweep()
        self.texts.sweep()


def is_inference(job: JobSnapshot) -> bool:
    """Tell encode and decode jobs apart from training-family jobs."""
    return job.operation in ("encode", "decode")


def interruption_error(job: JobSnapshot) -> dict[str, str] | None:
    """Explain lost secret inputs for inference jobs; keep other errors as they are."""
    return dict(INPUTS_LOST) if is_inference(job) else job.error


def require_inference(service: "WorkspaceJobService") -> InferenceServices:
    """Return the attached inference stores or explain that they are missing."""
    if service.inference is None:
        raise ApplicationFailure(
            "inference_unavailable",
            "Encode and Decode are not available in this application.",
        )
    return service.inference


def submit_inference(
    service: "WorkspaceJobService",
    request: EncodingJobRequest | DecodingJobRequest,
) -> JobSnapshot:
    """Accept one bounded job, saving settings durably and secrets in memory only."""
    service.require_storage()
    inference = require_inference(service)
    secrets = {"password": request.password}
    if isinstance(request, EncodingJobRequest):
        secrets["message"] = request.message
    fingerprint = service.store.fingerprint(
        {
            "operation": "encode" if "message" in secrets else "decode",
            "image_reference": request.image_reference,
            "model_identifier": request.model_identifier,
            "secret_digest": inference.secrets.digest(secrets),
        }
    )
    with service.lock:
        previous = service.store.replay(request.client_request_identifier, fingerprint)
        if previous is not None:
            return previous
        record = admit_inference(inference.files, inference.installed, request)
        if service.closing.is_set():
            raise ApplicationFailure(
                "service_stopping", "The application is stopping. Retry after restart."
            )
        waiting = [
            job
            for job in service.store.list_jobs()
            if is_inference(job) and job.status in ("queued", "running")
        ]
        if len(waiting) >= MAXIMUM_INFERENCE_JOBS:
            raise queue_full_failure()
        now = datetime.now(UTC)
        snapshot = JobSnapshot(
            job_identifier="job_" + uuid4().hex,
            status="queued",
            phase="queued",
            configuration=ConfigurationProfile(),
            available_actions=["cancel"],
            created_at=now,
            updated_at=now,
            operation=record.operation,
            experiment_identifier=None,
            frozen_settings=record.model_dump(mode="json"),
        )
        accepted = service.store.save(
            snapshot,
            request=record,
            mutation=(request.client_request_identifier, fingerprint),
        )
        inference.secrets.put(accepted.job_identifier, secrets)
        service.wake.set()
        return accepted
