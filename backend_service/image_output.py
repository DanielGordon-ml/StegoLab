"""Publish verified PNG pixels without replacing any existing output."""

import os
import tempfile
from collections.abc import Buffer, Callable
from io import BufferedIOBase
from pathlib import Path
from typing import BinaryIO, cast

from PIL import Image, ImageChops

from backend_service.image_diagnostics import private_image_diagnostics
from backend_service.image_policy import DATASET_PREPARED_POLICY
from backend_service.image_validation import image_failure


class BoundedPngWriter(BufferedIOBase):
    """Check each encoded write without exposing a descriptor that bypasses it."""

    def __init__(self, stream: BinaryIO, before_write: Callable[[int], None]) -> None:
        """Keep the output stream and cumulative encoded-byte allowance."""
        super().__init__()
        self.stream = stream
        self.before_write = before_write
        self.written = 0

    def write(self, buffer: Buffer) -> int:
        """Reserve each bounded chunk before touching the output file."""
        size = memoryview(buffer).nbytes
        if self.written + size > DATASET_PREPARED_POLICY.maximum_bytes:
            raise DATASET_PREPARED_POLICY.limit_failure()
        self.before_write(size)
        written = self.stream.write(buffer)
        if written != size:
            raise OSError("The output write was incomplete.")
        self.written += written
        return written

    def flush(self) -> None:
        """Flush while the owned wrapper and external stream remain open."""
        if not self.closed and not self.stream.closed:
            self.stream.flush()


def write_png(
    image: Image.Image,
    destination: Path,
    before_write: Callable[[int], None] | None = None,
) -> None:
    """Verify a same-directory temporary PNG, then publish with no overwrite."""
    try:
        with private_image_diagnostics():
            _publish_png(image, destination, before_write)
    except MemoryError:
        raise image_failure("image_resources") from None
    except (OSError, ValueError, TypeError):
        raise image_failure("image_write") from None


def _publish_png(
    image: Image.Image,
    destination: Path,
    before_write: Callable[[int], None] | None,
) -> None:
    """Create and verify the temporary output before no-overwrite publication."""
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=".stegolab-image-", suffix=".png", dir=destination.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            if before_write is None:
                image.save(stream, format="PNG")
            else:
                with BoundedPngWriter(stream, before_write) as bounded:
                    image.save(cast(BinaryIO, bounded), format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        with Image.open(temporary, formats=["PNG"]) as reopened:
            reopened.load()
            if reopened.mode != image.mode or reopened.size != image.size:
                raise image_failure("image_write")
            with ImageChops.difference(image, reopened) as difference:
                for band in image.getbands():
                    with difference.getchannel(band) as channel:
                        if channel.getextrema()[1]:
                            raise image_failure("image_write")
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
