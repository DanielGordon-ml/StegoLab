"""Encode and decode job submission and one-time reads of recovered text."""

from typing import cast

from fastapi import APIRouter, Request, Response

from backend_service.failures import ApplicationFailure
from backend_service.inference_jobs import require_inference, submit_inference
from backend_service.workspace_jobs import WorkspaceJobService
from schemas.errors import ErrorEnvelope
from schemas.inference_jobs import DecodedText, DecodingJobRequest, EncodingJobRequest
from schemas.jobs import JobSnapshot

inference_router = APIRouter(
    prefix="/api/v1",
    responses={
        status: {"model": ErrorEnvelope} for status in (404, 409, 422, 500, 503)
    },
)


def service(request: Request) -> WorkspaceJobService:
    """Read the one supervisor created during application startup."""
    return cast(WorkspaceJobService, request.app.state.workspace_jobs)


@inference_router.post(
    "/encoding_jobs",
    response_model=JobSnapshot,
    status_code=202,
    operation_id="submit_encoding_job",
)
def encoding(payload: EncodingJobRequest, request: Request) -> JobSnapshot:
    """Queue one encode of an uploaded cover; the secrets stay in memory."""
    snapshot = submit_inference(service(request), payload)
    request.app.state.event_logger.info("encoding_job_accepted")
    return snapshot


@inference_router.post(
    "/decoding_jobs",
    response_model=JobSnapshot,
    status_code=202,
    operation_id="submit_decoding_job",
)
def decoding(payload: DecodingJobRequest, request: Request) -> JobSnapshot:
    """Queue one decode of an uploaded encoded PNG; the password stays in memory."""
    snapshot = submit_inference(service(request), payload)
    request.app.state.event_logger.info("decoding_job_accepted")
    return snapshot


@inference_router.get(
    "/jobs/{job_identifier}/decoded_text",
    response_model=DecodedText,
    operation_id="read_decoded_text",
)
def read_decoded_text(
    job_identifier: str, request: Request, response: Response
) -> DecodedText:
    """Return recovered text from memory while it is still available."""
    supervisor = service(request)
    inference = require_inference(supervisor)
    job = supervisor.get_job(job_identifier)
    if job.operation != "decode":
        raise ApplicationFailure(
            "result_unavailable", "Only decode jobs produce recovered text.", 409
        )
    if job.status in ("queued", "running"):
        raise ApplicationFailure(
            "result_unavailable",
            "The recovered text is not ready. Wait for the decode job to finish.",
            409,
        )
    if job.status != "completed":
        raise ApplicationFailure(
            "result_unavailable",
            "This decode job did not produce text. Check its error and start it again.",
            409,
        )
    entry = inference.texts.get(job_identifier)
    if entry is None:
        raise ApplicationFailure(
            "result_expired",
            "The recovered text expired. Enter the password and decode again.",
            404,
        )
    response.headers["Cache-Control"] = "no-store"
    return entry


@inference_router.delete(
    "/jobs/{job_identifier}/decoded_text",
    status_code=204,
    operation_id="discard_decoded_text",
)
def discard_decoded_text(job_identifier: str, request: Request) -> Response:
    """Forget recovered text as soon as the reader is done, safe to repeat."""
    require_inference(service(request)).texts.discard(job_identifier)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
