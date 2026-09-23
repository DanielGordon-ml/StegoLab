"""Keep Pillow metadata logs private only inside image service calls."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_image_call_active: ContextVar[bool] = ContextVar("image_call_active", default=False)


class ImagePrivacyFilter(logging.Filter):
    """Suppress source-bearing Pillow records in the current service context."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Leave concurrent and unrelated Pillow logging unchanged."""
        return not _image_call_active.get()


for _logger_name in (
    "PIL.Image",
    "PIL.PngImagePlugin",
    "PIL.TiffImagePlugin",
    "PIL.JpegImagePlugin",
    "PIL.ImageCms",
):
    logging.getLogger(_logger_name).addFilter(ImagePrivacyFilter())


@contextmanager
def private_image_diagnostics() -> Iterator[None]:
    """Limit source metadata logging without changing global logger levels."""
    token = _image_call_active.set(True)
    try:
        yield
    finally:
        _image_call_active.reset(token)
