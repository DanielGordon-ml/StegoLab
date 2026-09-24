"""Available application features for this local foundation release."""

from typing import Annotated, Literal

from pydantic import Field, field_validator

from schemas.base import StrictRecord


class HealthStatus(StrictRecord):
    """Report successful application startup."""

    status: Literal["ready"] = "ready"
    application_version: Literal["0.1.0"] = "0.1.0"


class Capabilities(StrictRecord):
    """Separate experimental training from qualified inference support."""

    application_version: Literal["0.1.0"] = "0.1.0"
    available_devices: list[Literal["cpu"]] = Field(
        default=["cpu"], min_length=1, max_length=1
    )
    available_models: Annotated[list[str], Field(max_length=0)] = Field(
        default_factory=list
    )
    available_profiles: Annotated[list[str], Field(max_length=0)] = Field(
        default_factory=list
    )
    encoding_available: Literal[False] = False
    decoding_available: Literal[False] = False
    training_available: bool = True
    maximum_payload_bytes: Annotated[int, Field(ge=0, le=0)] = 0

    @field_validator(
        "encoding_available", "decoding_available", "training_available", mode="before"
    )
    @classmethod
    def validate_availability_type(cls, value: object) -> object:
        """Reject numeric values that compare equal to a boolean literal."""
        if type(value) is not bool:
            raise ValueError("Feature availability must be a boolean.")
        return value


class ModelList(StrictRecord):
    """Return an empty installed-model list until model support is available."""

    items: Annotated[list[str], Field(max_length=0)] = Field(default_factory=list)
