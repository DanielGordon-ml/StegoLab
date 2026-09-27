"""Chunked dataset archive uploads that a fetch job can import as a source."""

from typing import cast

from fastapi import APIRouter, Request, Response
from starlette.concurrency import run_in_threadpool

from backend_service.dataset_sources.upload_sessions import DatasetUploadStore
from backend_service.failures import ApplicationFailure
from backend_service.request_bodies import read_bounded_body
from schemas.dataset_uploads import (
    DATASET_UPLOAD_CHUNK_BYTES,
    DatasetUploadChunk,
    DatasetUploadCompleteRequest,
    DatasetUploadCreateRequest,
    DatasetUploadSession,
)
from schemas.errors import ErrorEnvelope

dataset_upload_router = APIRouter(
    prefix="/api/v1",
    responses={
        status: {"model": ErrorEnvelope}
        for status in (404, 409, 413, 422, 500, 503, 507)
    },
)
CHUNK_BODY = {
    "requestBody": {
        "required": True,
        "description": "One archive part: exactly the part size, at most 16 MiB.",
        "content": {
            "application/octet-stream": {
                "schema": {"type": "string", "format": "binary"}
            }
        },
    }
}
_MESSAGES: dict[str, tuple[str, int]] = {
    "upload_too_large": (
        "The part is larger than the 16 MiB part limit. Send the archive in "
        "parts of at most 16 MiB.",
        413,
    ),
}


def upload_body_failure(code: str) -> ApplicationFailure:
    """Return the fixed failure for a problem with an upload request body."""
    message, status_code = _MESSAGES[code]
    return ApplicationFailure(code, message, status_code)


def upload_store(request: Request) -> DatasetUploadStore:
    """Read the upload store created during application startup."""
    return cast(DatasetUploadStore, request.app.state.dataset_uploads)


@dataset_upload_router.post(
    "/datasets/uploads",
    response_model=DatasetUploadSession,
    status_code=201,
    operation_id="create_dataset_upload",
)
def create_dataset_upload(
    payload: DatasetUploadCreateRequest, request: Request
) -> DatasetUploadSession:
    """Open an upload session for one archive; a repeated request replays it."""
    session = upload_store(request).create(payload)
    request.app.state.event_logger.info("dataset_upload_created")
    return session


@dataset_upload_router.put(
    "/datasets/uploads/{upload_identifier}/chunks/{chunk_index}",
    response_model=DatasetUploadChunk,
    operation_id="upload_dataset_chunk",
    openapi_extra=CHUNK_BODY,
)
async def upload_dataset_chunk(
    upload_identifier: str, chunk_index: int, request: Request
) -> DatasetUploadChunk:
    """Store one raw part of exactly the expected size; parts arrive in any order."""
    data = await read_bounded_body(
        request,
        DATASET_UPLOAD_CHUNK_BYTES,
        lambda: upload_body_failure("upload_too_large"),
    )
    record = await run_in_threadpool(
        upload_store(request).receive_chunk, upload_identifier, chunk_index, data
    )
    request.app.state.event_logger.info("dataset_upload_chunk_received")
    return record


@dataset_upload_router.get(
    "/datasets/uploads/{upload_identifier}",
    response_model=DatasetUploadSession,
    operation_id="read_dataset_upload",
)
def read_dataset_upload(
    upload_identifier: str, request: Request
) -> DatasetUploadSession:
    """Report which parts arrived so a reloaded browser can continue."""
    return upload_store(request).read(upload_identifier)


@dataset_upload_router.post(
    "/datasets/uploads/{upload_identifier}/complete",
    response_model=DatasetUploadSession,
    operation_id="complete_dataset_upload",
)
def complete_dataset_upload(
    upload_identifier: str, payload: DatasetUploadCompleteRequest, request: Request
) -> DatasetUploadSession:
    """Join every part into the archive and verify it; repeating is harmless."""
    session = upload_store(request).complete(upload_identifier)
    request.app.state.event_logger.info("dataset_upload_completed")
    return session


@dataset_upload_router.delete(
    "/datasets/uploads/{upload_identifier}",
    status_code=204,
    operation_id="discard_dataset_upload",
)
def discard_dataset_upload(upload_identifier: str, request: Request) -> Response:
    """Remove an upload and everything it stored; repeating is harmless."""
    upload_store(request).discard(upload_identifier)
    request.app.state.event_logger.info("dataset_upload_discarded")
    return Response(status_code=204)
