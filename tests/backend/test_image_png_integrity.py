"""Reject CRC-valid malformed PNG ordering and compressed pixel streams."""

import struct
import zlib
from io import BytesIO

import pytest
from PIL import Image

from backend_service.failures import ApplicationFailure
from backend_service.image_preparation import inspect_image, read_prepared_png
from backend_service.image_validation import PNG_SIGNATURE


def chunk(name: bytes, payload: bytes) -> bytes:
    """Encode a valid chunk checksum independently of image validation."""
    return (
        struct.pack(">I", len(payload))
        + name
        + payload
        + struct.pack(">I", zlib.crc32(name + payload))
    )


def png_with_chunks(chunks: list[tuple[bytes, bytes]]) -> bytes:
    """Use fixed RGB dimensions and independently assembled PNG chunks."""
    header = struct.pack(">IIBBBBB", 512, 512, 8, 2, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", header)
        + b"".join(chunk(name, payload) for name, payload in chunks)
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize(
    "damage", ["late_transparency", "late_palette", "duplicate_palette", "split_pixels"]
)
def test_malformed_chunk_order_rejected_before_decode(
    damage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject dropped transparency and malformed palette or pixel placement."""
    compressed = zlib.compress((b"\0" + b"\x32\x41\x58" * 512) * 512)
    palette = (b"PLTE", b"\0\0\0")
    if damage == "late_transparency":
        chunks = [(b"IDAT", compressed), (b"tRNS", struct.pack(">HHH", 50, 65, 88))]
    elif damage == "late_palette":
        chunks = [(b"IDAT", compressed), palette]
    elif damage == "duplicate_palette":
        chunks = [palette, palette, (b"IDAT", compressed)]
    else:
        chunks = [
            (b"IDAT", compressed[:10]),
            (b"tEXt", b"note\0value"),
            (b"IDAT", compressed[10:]),
        ]

    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail if malformed structure is handed to Pillow."""
        pytest.fail("Malformed PNG must fail before decoding.")

    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(ApplicationFailure):
        inspect_image(BytesIO(png_with_chunks(chunks)))


@pytest.mark.parametrize(
    "damage",
    [
        "missing_checksum",
        "bad_checksum",
        "extra_stream",
        "short_pixels",
        "too_many_pixels",
    ],
)
def test_incomplete_or_excess_pixel_stream_is_rejected(damage: str) -> None:
    """Require zlib end, a correct checksum, and exactly the scanline budget."""
    pixels = (b"\0" + b"\x32\x41\x58" * 512) * 512
    compressed = zlib.compress(pixels)
    if damage == "missing_checksum":
        compressed = compressed[:-4]
    elif damage == "bad_checksum":
        compressed = compressed[:-4] + b"\0" * 4
    elif damage == "extra_stream":
        compressed += zlib.compress(b"second-stream")
    elif damage == "short_pixels":
        compressed = zlib.compress(pixels[:-1])
    else:
        compressed = zlib.compress(pixels + b"\0" * 1_000_000)
    with pytest.raises(ApplicationFailure):
        inspect_image(BytesIO(png_with_chunks([(b"IDAT", compressed)])))


def test_valid_palette_and_consecutive_pixel_chunks_are_accepted() -> None:
    """Retain allowed truecolor palettes and arbitrary consecutive IDAT splits."""
    compressed = zlib.compress((b"\0" + b"\x32\x41\x58" * 512) * 512)
    chunks = [(b"PLTE", b"\0\0\0")]
    chunks += [
        (b"IDAT", compressed[start : start + 13])
        for start in range(0, len(compressed), 13)
    ]
    chunks.append((b"IDAT", b""))
    assert inspect_image(BytesIO(png_with_chunks(chunks))).source_width == 512


def test_real_adam7_pixels_are_accepted_without_alteration() -> None:
    """Use independent fixed pass dimensions for a 513 by 515 RGB fixture."""
    pass_dimensions = [
        (65, 65),
        (64, 65),
        (129, 64),
        (128, 129),
        (257, 129),
        (256, 258),
        (513, 257),
    ]
    pixels = b"".join(
        (b"\0" + b"\x32\x41\x58" * width) * height for width, height in pass_dimensions
    )
    header = struct.pack(">IIBBBBB", 513, 515, 8, 2, 0, 0, 1)
    source = (
        PNG_SIGNATURE
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    recovered = read_prepared_png(BytesIO(source))
    assert recovered.image.size == (513, 515)
    assert recovered.image.getextrema() == ((50, 50), (65, 65), (88, 88))
