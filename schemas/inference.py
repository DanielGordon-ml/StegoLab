"""Uploaded image records and capacity checks for experimental encode and decode."""

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from schemas.base import StrictRecord
from schemas.images import ImageSummary
from schemas.models import ModelIdentifier
from schemas.protocol import PayloadCapacity

ImageReference = Annotated[str, Field(pattern=r"^image_[0-9a-f]{32}$")]
StoredReference = Annotated[str, Field(pattern=r"^(image|encoded)_[0-9a-f]{32}$")]
ImagePurpose = Literal["cover", "encoded"]
MAXIMUM_STORED_IMAGE_BYTES = 64 * 1024 * 1024
RETENTION_SECONDS = 24 * 60 * 60


class UploadedImage(StrictRecord):
    """Describe one accepted upload without its file name, path, or bytes."""

    image_reference: ImageReference
    purpose: ImagePurpose
    summary: ImageSummary
    warnings: list[str] = Field(default_factory=list, max_length=4)
    created_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def validate_purpose_and_expiry(self) -> Self:
        """Tie the pixel policy to the purpose and keep the expiry after creation."""
        expected = "prepare_srgb" if self.purpose == "cover" else "preserve_stored"
        if self.summary.pixel_policy != expected:
            raise ValueError("The image summary does not match the upload purpose.")
        if self.expires_at <= self.created_at:
            raise ValueError("The image expiry must be after its creation time.")
        return self


class StoredImageRecord(UploadedImage):
    """Persist an upload or a published result with its stored PNG checksum."""

    image_reference: StoredReference
    stored_bytes: int = Field(ge=1, le=MAXIMUM_STORED_IMAGE_BYTES)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    def public(self) -> UploadedImage:
        """Return the client-facing record without storage details."""
        return UploadedImage.model_validate(
            self.model_dump(exclude={"stored_bytes", "checksum"})
        )


class CapacityRequest(StrictRecord):
    """Ask how many message bytes one uploaded image can carry with one model."""

    image_reference: ImageReference
    model_identifier: ModelIdentifier


class CapacityResult(StrictRecord):
    """Report the message limit for one image and model inside the model range."""

    image_reference: ImageReference
    model_identifier: ModelIdentifier
    profile_identifier: Literal["test_only_v1"] = "test_only_v1"
    width: int = Field(ge=512, le=1024)
    height: int = Field(ge=512, le=1024)
    maximum_message_bytes: int = Field(ge=256, le=1024)
    capacity: PayloadCapacity

    @model_validator(mode="after")
    def validate_limit_matches_layout(self) -> Self:
        """Keep the advertised limit equal to the detailed layout accounting."""
        if self.maximum_message_bytes != self.capacity.maximum_message_bytes:
            raise ValueError("The message limit must match the layout capacity.")
        if self.capacity.payload_map_bits != self.width * self.height:
            raise ValueError("The layout capacity must match the image size.")
        return self
