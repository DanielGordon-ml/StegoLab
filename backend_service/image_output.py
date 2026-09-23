"""Publish verified PNG pixels without replacing any existing output."""

import os
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

from backend_service.image_validation import image_failure


def write_png(image: Image.Image, destination: Path) -> None:
    """Verify a same-directory temporary PNG, then publish with no overwrite."""
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=".stegolab-image-", suffix=".png", dir=destination.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            image.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        with Image.open(temporary) as reopened:
            reopened.load()
            if reopened.mode != image.mode or reopened.size != image.size:
                raise image_failure("image_write")
            difference = ImageChops.difference(image, reopened)
            if any(
                difference.getchannel(band).getextrema()[1] for band in image.getbands()
            ):
                raise image_failure("image_write")
        os.link(temporary, destination)
    except (OSError, ValueError, TypeError):
        raise image_failure("image_write") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
