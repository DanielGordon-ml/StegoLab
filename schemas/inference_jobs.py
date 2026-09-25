"""Encode and decode job requests, persisted records, and results without secrets."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field

from schemas.base import StrictRecord
from schemas.configuration import RequestIdentifier
from schemas.inference import ImageReference
from schemas.models import ModelIdentifier

MAXIMUM_MESSAGE_BYTES = 1024
MAXIMUM_PASSWORD_LENGTH = 1024
DECODED_TEXT_LIFETIME_SECONDS = 300
MAXIMUM_INFERENCE_JOBS = 8

ArtifactReference = Annotated[str, Field(pattern=r"^encoded_[0-9a-f]{32}$")]


class EncodingJobRequest(StrictRecord):
    """Transport-only encode request; the message and password are never stored."""

    client_request_identifier: RequestIdentifier
    image_reference: ImageReference
    model_identifier: ModelIdentifier
    message: str = Field(max_length=MAXIMUM_MESSAGE_BYTES)
    password: str = Field(min_length=1, max_length=MAXIMUM_PASSWORD_LENGTH)


class DecodingJobRequest(StrictRecord):
    """Transport-only decode request; the password is never stored."""

    client_request_identifier: RequestIdentifier
    image_reference: ImageReference
    model_identifier: ModelIdentifier
    password: str = Field(min_length=1, max_length=MAXIMUM_PASSWORD_LENGTH)


class InferenceJobRecord(StrictRecord):
    """Persisted job settings; a password or message key can never validate here."""

    operation: Literal["encode", "decode"]
    image_reference: ImageReference
    model_identifier: ModelIdentifier
    message_byte_count: int = Field(ge=0, le=MAXIMUM_MESSAGE_BYTES)
    width: int = Field(ge=512, le=1024)
    height: int = Field(ge=512, le=1024)


class EncodingJobResult(StrictRecord):
    """Describe a published, decoder-verified PNG without any message content."""

    artifact_identifier: ArtifactReference
    filename: str = Field(pattern=r"^stegolab-encoded-[0-9a-f]{8}\.png$")
    width: int = Field(ge=512, le=1024)
    height: int = Field(ge=512, le=1024)
    png_bytes: int = Field(ge=1)
    message_byte_count: int = Field(ge=0, le=MAXIMUM_MESSAGE_BYTES)
    verified: Literal[True] = True


class DecodingJobResult(StrictRecord):
    """Announce recovered text held in memory for a short time, never its content."""

    result_available: Literal[True] = True
    text_byte_count: int = Field(ge=0, le=MAXIMUM_MESSAGE_BYTES)
    expires_at: AwareDatetime


class DecodedText(StrictRecord):
    """Return recovered text once per request; it is never written to disk."""

    job_identifier: str = Field(min_length=1, max_length=128)
    text: str = Field(max_length=MAXIMUM_MESSAGE_BYTES)
    byte_count: int = Field(ge=0, le=MAXIMUM_MESSAGE_BYTES)
    expires_at: AwareDatetime
