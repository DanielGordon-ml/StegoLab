"""Admission checks shared by capacity requests and experimental inference jobs."""

from backend_service.failures import ApplicationFailure
from schemas.models import InstalledModel


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
