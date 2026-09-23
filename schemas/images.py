"""Safe metadata for validated image preparation."""

from typing import Literal, Self

from pydantic import Field, model_validator

from schemas.base import StrictRecord
from schemas.image_dimensions import ImageDimensions

MAXIMUM_IMAGE_FILE_BYTES = 50 * 1024 * 1024


class ImageSummary(StrictRecord):
    """Describe image handling without paths or source metadata."""

    source_format: Literal["JPEG", "PNG"]
    mode: Literal["RGB", "RGBA"]
    pixel_policy: Literal["prepare_srgb", "preserve_stored"]
    source_width: int
    source_height: int
    prepared_width: int
    prepared_height: int
    input_size_bytes: int = Field(ge=1, le=MAXIMUM_IMAGE_FILE_BYTES)
    orientation: int = Field(ge=1, le=8)
    color_policy: Literal["icc_to_srgb", "declared_srgb", "assumed_srgb"]

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """Use the shared limits for original and prepared dimensions."""
        ImageDimensions(width=self.source_width, height=self.source_height)
        ImageDimensions(width=self.prepared_width, height=self.prepared_height)
        return self
