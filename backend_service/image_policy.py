"""Trusted image limits shared by preparation and dataset validation."""

from dataclasses import dataclass

from backend_service.failures import ApplicationFailure


@dataclass(frozen=True)
class ImagePolicy:
    """Keep internal allocation limits separate from public image contracts."""

    minimum_side: int
    maximum_side: int
    maximum_pixels: int
    maximum_bytes: int
    allow_grayscale: bool = False
    dataset: bool = False

    def check_dimensions(self, width: int, height: int) -> None:
        """Reject dimensions before allocating decoded pixel buffers."""
        if not (
            type(width) is int
            and type(height) is int
            and self.minimum_side <= width <= self.maximum_side
            and self.minimum_side <= height <= self.maximum_side
            and width * height <= self.maximum_pixels
        ):
            raise self.limit_failure()

    def limit_failure(self) -> ApplicationFailure:
        """Describe the applicable fixed policy without submitted values."""
        if self.dataset:
            message = (
                "The dataset image exceeds supported limits. Use sides of "
                "1–8192 pixels, at most 32,000,000 pixels, and at most "
                f"{self.maximum_bytes // (1024 * 1024)} MiB."
            )
        else:
            message = (
                "The image exceeds supported limits. Use sides of "
                "512–4096 pixels, at most 8,850,000 pixels, and at most 50 MiB."
            )
        return ApplicationFailure("image_limits", message, status_code=422)


PRODUCTION_IMAGE_POLICY = ImagePolicy(512, 4096, 8_850_000, 50 * 1024 * 1024)
DATASET_SOURCE_POLICY = ImagePolicy(
    1, 8192, 32_000_000, 50 * 1024 * 1024, allow_grayscale=True, dataset=True
)
DATASET_PREPARED_POLICY = ImagePolicy(
    1, 8192, 32_000_000, 128 * 1024 * 1024, dataset=True
)
