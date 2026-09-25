"""Image uploads and capacity checks for experimental encode and decode."""

from typing import Annotated, Literal, cast

from fastapi import APIRouter, Query, Request
from starlette.concurrency import run_in_threadpool

from backend_service.inference_admission import check_model_range
from backend_service.inference_files import InferenceFileStore
from backend_service.inference_storage import upload_failure
from backend_service.model_installation import InstalledModelStore
from backend_service.payload_capacity import calculate_capacity
from schemas.capabilities import MAXIMUM_UPLOAD_BYTES
from schemas.errors import ErrorEnvelope
from schemas.inference import CapacityRequest, CapacityResult, UploadedImage
from schemas.protocol import ProtocolContext

image_router = APIRouter(
    prefix="/api/v1",
    responses={
        404: {"model": ErrorEnvelope},
        413: {"model": ErrorEnvelope},
        415: {"model": ErrorEnvelope},
        422: {"model": ErrorEnvelope},
        503: {"model": ErrorEnvelope},
    },
)

UPLOAD_CONTENT_TYPES = ("image/png", "image/jpeg", "application/octet-stream")
UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "description": "The image file itself, at most 16 MiB.",
        "content": {
            "image/png": {"schema": {"type": "string", "format": "binary"}},
            "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
        },
    }
}


def inference_files(request: Request) -> InferenceFileStore:
    """Read the upload store created during application startup."""
    return cast(InferenceFileStore, request.app.state.inference_files)


async def read_bounded_body(request: Request) -> bytes:
    """Collect the raw body while refusing anything above the upload limit."""
    declared = request.headers.get("content-length")
    if declared is not None and (
        not declared.isdigit() or int(declared) > MAXIMUM_UPLOAD_BYTES
    ):
        raise upload_failure("upload_too_large")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAXIMUM_UPLOAD_BYTES:
            raise upload_failure("upload_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


@image_router.post(
    "/images",
    response_model=UploadedImage,
    status_code=201,
    operation_id="upload_image",
    openapi_extra=UPLOAD_BODY,
)
async def upload_image(
    request: Request, purpose: Annotated[Literal["cover", "encoded"], Query()]
) -> UploadedImage:
    """Store one raw image body for encoding (cover) or decoding (encoded)."""
    media_type = request.headers.get("content-type", "").split(";")[0].strip()
    if media_type.lower() not in UPLOAD_CONTENT_TYPES:
        raise upload_failure("upload_content_type")
    data = await read_bounded_body(request)
    record = await run_in_threadpool(inference_files(request).store, purpose, data)
    request.app.state.event_logger.info("image_upload_completed")
    return record


@image_router.post(
    "/capacity", response_model=CapacityResult, operation_id="read_capacity"
)
def read_capacity(payload: CapacityRequest, request: Request) -> CapacityResult:
    """Report how many message bytes one uploaded image carries with one model."""
    record = inference_files(request).read_record(payload.image_reference)
    installed = cast(InstalledModelStore, request.app.state.installed_models)
    model, _ = installed.resolve(payload.model_identifier)
    width, height = record.summary.prepared_width, record.summary.prepared_height
    check_model_range(model, width, height)
    capacity = calculate_capacity(
        ProtocolContext(
            width=width,
            height=height,
            compatibility_identifier=model.compatibility_identifier,
        )
    )
    return CapacityResult(
        image_reference=payload.image_reference,
        model_identifier=payload.model_identifier,
        width=width,
        height=height,
        maximum_message_bytes=capacity.maximum_message_bytes,
        capacity=capacity,
    )
