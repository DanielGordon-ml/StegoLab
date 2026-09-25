"""Chunked, resumable dataset archive uploads with server-side checksums."""

import math
from typing import Final, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from schemas.base import StrictRecord
from schemas.configuration import RequestIdentifier
from schemas.dataset_common import SHA256
from schemas.dataset_sources import UploadReference

MAXIMUM_DATASET_UPLOAD_BYTES: Final = 2 * 1024**3
DATASET_UPLOAD_CHUNK_BYTES: Final = 16 * 1024**2
MAXIMUM_UPLOAD_CHUNKS: Final = 128
UPLOAD_RETENTION_SECONDS: Final = 24 * 60 * 60


class DatasetUploadCreateRequest(StrictRecord):
    """Open an upload session for one archive of a known total size."""

    client_request_identifier: RequestIdentifier
    file_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,254}$")
    total_bytes: int = Field(ge=1, le=MAXIMUM_DATASET_UPLOAD_BYTES)
    expected_sha256: SHA256 | None = None


class DatasetUploadSession(StrictRecord):
    """Report which parts arrived so a reloaded browser can continue."""

    upload_identifier: UploadReference
    file_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,254}$")
    total_bytes: int = Field(ge=1, le=MAXIMUM_DATASET_UPLOAD_BYTES)
    chunk_bytes: Literal[16_777_216] = DATASET_UPLOAD_CHUNK_BYTES
    chunk_count: int = Field(ge=1, le=MAXIMUM_UPLOAD_CHUNKS)
    received_chunks: list[int] = Field(
        default_factory=list, max_length=MAXIMUM_UPLOAD_CHUNKS
    )
    complete: bool = False
    sha256: SHA256 | None = None
    created_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def validate_parts(self) -> Self:
        """Keep the part list consistent with the declared size and state."""
        if self.chunk_count != math.ceil(self.total_bytes / self.chunk_bytes):
            raise ValueError("The part count does not match the upload size.")
        if self.received_chunks != sorted(set(self.received_chunks)) or any(
            index < 0 or index >= self.chunk_count for index in self.received_chunks
        ):
            raise ValueError("Received parts must be unique indexes in range.")
        if self.complete != (
            len(self.received_chunks) == self.chunk_count and self.sha256 is not None
        ):
            raise ValueError("A complete upload needs every part and a checksum.")
        return self


class DatasetUploadChunk(StrictRecord):
    """Acknowledge one received part with the digest the server computed."""

    upload_identifier: UploadReference
    index: int = Field(ge=0, lt=MAXIMUM_UPLOAD_CHUNKS)
    bytes: int = Field(ge=1, le=DATASET_UPLOAD_CHUNK_BYTES)
    sha256: SHA256
    received_chunks: list[int] = Field(
        default_factory=list, max_length=MAXIMUM_UPLOAD_CHUNKS
    )


class DatasetUploadCompleteRequest(StrictRecord):
    """Ask the server to assemble and verify every received part."""

    client_request_identifier: RequestIdentifier
