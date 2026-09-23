"""Safe error records that do not include submitted values."""

from pydantic import Field

from schemas.base import StrictRecord


class ApplicationError(StrictRecord):
    """Explain a failure and provide an opaque reference for local logs."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    diagnostic_reference: str = Field(pattern=r"^[a-f0-9]{32}$")


class ErrorEnvelope(StrictRecord):
    """Keep all application errors in one predictable response envelope."""

    error: ApplicationError
