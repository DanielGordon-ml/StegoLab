"""Admission checks shared by capacity requests and experimental inference jobs."""

from typing import Literal

from backend_service.failures import ApplicationFailure
from backend_service.inference_files import InferenceFileStore
from backend_service.message_frame import password_bytes
from backend_service.model_installation import InstalledModelStore
from backend_service.payload_capacity import calculate_capacity
from schemas.inference_jobs import (
    DecodingJobRequest,
    EncodingJobRequest,
    InferenceJobRecord,
)
from schemas.models import InstalledModel
from schemas.protocol import ProtocolContext


def model_range_failure(model: InstalledModel) -> ApplicationFailure:
    """Explain the installed model's side limits without repeating image values."""
    return ApplicationFailure(
        "image_outside_model_range",
        "This experimental model accepts images with sides between "
        f"{model.minimum_side} and {model.maximum_side} pixels and files up to "
        "16 MiB. Choose a smaller image or crop it before uploading.",
        422,
    )


def check_model_range(model: InstalledModel, width: int, height: int) -> None:
    """Refuse prepared dimensions outside the installed model's side range."""
    sides = (width, height)
    if not all(model.minimum_side <= side <= model.maximum_side for side in sides):
        raise model_range_failure(model)


def message_bytes(message: str, limit: int) -> int:
    """Count strict UTF-8 message bytes and refuse messages above the image limit."""
    try:
        count = len(message.encode("utf-8", errors="strict"))
    except UnicodeEncodeError:
        raise ApplicationFailure(
            "invalid_message",
            "The message contains characters that cannot be saved as UTF-8 text. "
            "Remove them and retry.",
            422,
        ) from None
    if count > limit:
        raise ApplicationFailure(
            "message_too_long",
            "The message is longer than this image can carry. Shorten it to at "
            f"most {limit} bytes of UTF-8 text.",
            422,
        )
    return count


def admit_inference(
    files: InferenceFileStore,
    installed: InstalledModelStore,
    request: EncodingJobRequest | DecodingJobRequest,
) -> InferenceJobRecord:
    """Check image, model, range, and secret sizes before a job is accepted."""
    operation: Literal["encode", "decode"] = (
        "encode" if isinstance(request, EncodingJobRequest) else "decode"
    )
    record = files.read_record(request.image_reference)
    if record.purpose != ("cover" if operation == "encode" else "encoded"):
        raise ApplicationFailure(
            "image_purpose_mismatch",
            "Encode needs an image uploaded as a cover, and Decode needs an image "
            "uploaded as encoded. Upload the file again with the right purpose.",
            422,
        )
    model, _ = installed.resolve(request.model_identifier)
    width, height = record.summary.prepared_width, record.summary.prepared_height
    check_model_range(model, width, height)
    password_bytes(request.password)
    message_byte_count = 0
    if isinstance(request, EncodingJobRequest):
        capacity = calculate_capacity(
            ProtocolContext(
                width=width,
                height=height,
                compatibility_identifier=model.compatibility_identifier,
            )
        )
        message_byte_count = message_bytes(
            request.message, capacity.maximum_message_bytes
        )
    return InferenceJobRecord(
        operation=operation,
        image_reference=request.image_reference,
        model_identifier=request.model_identifier,
        message_byte_count=message_byte_count,
        width=width,
        height=height,
    )
