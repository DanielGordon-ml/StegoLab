"""Available application features, derived from explicitly installed models."""

from typing import Final, Literal, Self

from pydantic import Field, field_validator, model_validator

from schemas.base import StrictRecord
from schemas.models import InstalledModel, ModelIdentifier

MAXIMUM_UPLOAD_BYTES: Final = 16_777_216


class HealthStatus(StrictRecord):
    """Report successful application startup."""

    status: Literal["ready"] = "ready"
    application_version: Literal["0.1.0"] = "0.1.0"


class Capabilities(StrictRecord):
    """Advertise installed experimental models separately from release quality."""

    application_version: Literal["0.1.0"] = "0.1.0"
    available_devices: list[Literal["cpu"]] = Field(
        default=["cpu"], min_length=1, max_length=1
    )
    available_models: list[ModelIdentifier] = Field(default_factory=list, max_length=8)
    available_profiles: list[Literal["test_only_v1"]] = Field(
        default_factory=list, max_length=1
    )
    encoding_available: bool = False
    decoding_available: bool = False
    training_available: bool = True
    experimental_models_only: Literal[True] = True
    maximum_payload_bytes: int = Field(default=0, ge=0, le=1024)
    minimum_image_side: int = Field(default=0, ge=0, le=4096)
    maximum_image_side: int = Field(default=0, ge=0, le=4096)
    maximum_upload_bytes: Literal[16_777_216] = 16_777_216

    @field_validator(
        "encoding_available", "decoding_available", "training_available", mode="before"
    )
    @classmethod
    def validate_availability_type(cls, value: object) -> object:
        """Reject numeric values that compare equal to a boolean literal."""
        if type(value) is not bool:
            raise ValueError("Feature availability must be a boolean.")
        return value

    @model_validator(mode="after")
    def consistent_with_installed_models(self) -> Self:
        """Reject availability claims that disagree with the installed-model list."""
        installed = bool(self.available_models)
        if (
            self.encoding_available != installed
            or self.decoding_available != installed
            or bool(self.available_profiles) != installed
            or (self.maximum_payload_bytes > 0) != installed
            or (self.minimum_image_side > 0) != installed
            or (self.maximum_image_side > 0) != installed
            or (installed and self.minimum_image_side > self.maximum_image_side)
        ):
            raise ValueError("Capabilities must match the installed model list.")
        return self


class ModelList(StrictRecord):
    """List explicitly installed experimental models."""

    items: list[InstalledModel] = Field(default_factory=list, max_length=8)
