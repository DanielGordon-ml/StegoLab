"""Inspect and prepare local images with explicit color and pixel rules."""

import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from struct import error as struct_error

from PIL import Image, ImageOps

from backend_service.failures import ApplicationFailure
from backend_service.image_color import convert_color, resolve_color
from backend_service.image_diagnostics import private_image_diagnostics
from backend_service.image_output import write_png
from backend_service.image_validation import (
    ImageSource,
    SourceHeader,
    image_failure,
    inspect_header,
    read_source,
)
from schemas.images import ImageSummary


@dataclass(frozen=True)
class PreparedPixels:
    """Return owned integer pixels and safe metadata without source metadata."""

    image: Image.Image
    summary: ImageSummary


def _open_pixels(data: bytes, header: SourceHeader) -> Image.Image:
    """Decode only after source precision and allocation limits were checked."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data), formats=["PNG", "JPEG"]) as opened:
            if (
                opened.format != header.format
                or opened.mode != header.mode
                or opened.size != (header.width, header.height)
                or getattr(opened, "n_frames", 1) != 1
                or "transparency" in opened.info
            ):
                raise image_failure()
            opened.load()
            return opened.copy()


def _decode_pixels(
    data: bytes, header: SourceHeader, *, recover: bool
) -> PreparedPixels:
    """Prepare validated source pixels or keep recovery pixels unchanged."""
    if recover and header.format != "PNG":
        raise image_failure()
    original = _open_pixels(data, header)
    orientation = original.getexif().get(274, 1)
    if type(orientation) is not int or not 1 <= orientation <= 8:
        raise image_failure()
    color_policy, profile = resolve_color(original, header)
    if recover:
        prepared = original
    else:
        oriented = ImageOps.exif_transpose(original)
        prepared = convert_color(oriented, profile)
    # Rebuild from integer pixels so no source metadata or profile survives.
    clean = Image.new(prepared.mode, prepared.size)
    clean.paste(prepared)
    summary = ImageSummary(
        source_format=header.format,
        mode=header.mode,
        pixel_policy="preserve_stored" if recover else "prepare_srgb",
        source_width=header.width,
        source_height=header.height,
        prepared_width=clean.width,
        prepared_height=clean.height,
        input_size_bytes=len(data),
        orientation=orientation,
        color_policy=color_policy,
    )
    return PreparedPixels(clean, summary)


def _read_pixels(source: ImageSource, *, recover: bool) -> PreparedPixels:
    """Sanitize parsing failures and metadata warnings at the service boundary."""
    try:
        data = read_source(source)
        header = inspect_header(data)
        with private_image_diagnostics(), warnings.catch_warnings():
            warnings.simplefilter("error")
            return _decode_pixels(data, header, recover=recover)
    except ApplicationFailure:
        raise
    except MemoryError:
        raise image_failure("image_resources") from None
    except (
        OSError,
        ValueError,
        TypeError,
        SyntaxError,
        KeyError,
        IndexError,
        struct_error,
        Image.DecompressionBombError,
        Warning,
    ):
        raise image_failure() from None


def inspect_image(source: ImageSource) -> ImageSummary:
    """Validate the complete input and report the planned preparation."""
    return _read_pixels(source, recover=False).summary


def prepare_image(source: ImageSource, destination: Path) -> ImageSummary:
    """Prepare sRGB pixels and save a metadata-free PNG to a new output path."""
    prepared = _read_pixels(source, recover=False)
    write_png(prepared.image, destination)
    return prepared.summary


def read_prepared_png(source: ImageSource) -> PreparedPixels:
    """Read stored PNG integers without orientation or color conversion."""
    return _read_pixels(source, recover=True)
