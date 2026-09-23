"""Shared limits for prepared images and experimental payload maps."""

from typing import Self

from pydantic import Field, model_validator

from schemas.base import StrictRecord

MINIMUM_IMAGE_SIDE = 512
MAXIMUM_IMAGE_SIDE = 4096
MAXIMUM_IMAGE_AREA = 8_850_000


class ImageDimensions(StrictRecord):
    """Require supported integer dimensions before allocating image buffers."""

    width: int = Field(ge=MINIMUM_IMAGE_SIDE, le=MAXIMUM_IMAGE_SIDE)
    height: int = Field(ge=MINIMUM_IMAGE_SIDE, le=MAXIMUM_IMAGE_SIDE)

    @model_validator(mode="after")
    def validate_area(self) -> Self:
        """Keep both common 4K shapes within a bounded pixel count."""
        if self.width * self.height > MAXIMUM_IMAGE_AREA:
            raise ValueError("The image exceeds the supported pixel count.")
        return self
