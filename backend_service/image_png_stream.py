"""Validate PNG pixel compression with a bounded discarded output buffer."""

import zlib


class PixelStreamValidator:
    """Check one complete zlib stream and the exact stored scanline byte count."""

    def __init__(self, width: int, height: int, channels: int, interlace: int) -> None:
        """Calculate the pixel budget for ordinary or seven-pass Adam7 images."""
        passes = (
            [(0, 0, 1, 1)]
            if interlace == 0
            else [
                (0, 0, 8, 8),
                (4, 0, 8, 8),
                (0, 4, 4, 8),
                (2, 0, 4, 4),
                (0, 2, 2, 4),
                (1, 0, 2, 2),
                (0, 1, 1, 2),
            ]
        )
        self._expected_bytes = 0
        for start_x, start_y, step_x, step_y in passes:
            pass_width = max(0, (width - start_x + step_x - 1) // step_x)
            pass_height = max(0, (height - start_y + step_y - 1) // step_y)
            if pass_width and pass_height:
                self._expected_bytes += pass_height * (1 + pass_width * channels)
        self._decoded_bytes = 0
        self._decoder = zlib.decompressobj()

    def feed(self, data: bytes) -> None:
        """Discard bounded output while checking the checksum and total size."""
        if self._decoder.eof and data:
            raise ValueError("PNG contains data after its compressed pixel stream.")
        pending = data
        while True:
            output = self._decoder.decompress(pending, 64 * 1024)
            self._decoded_bytes += len(output)
            if self._decoded_bytes > self._expected_bytes or self._decoder.unused_data:
                raise ValueError("PNG pixel data exceeds its expected size.")
            pending = self._decoder.unconsumed_tail
            if not pending and len(output) < 64 * 1024:
                break

    def finish(self) -> None:
        """Reject incomplete checksums, short pixels, and unfinished streams."""
        if not self._decoder.eof or self._decoded_bytes != self._expected_bytes:
            raise ValueError("PNG compressed pixel data is incomplete.")
