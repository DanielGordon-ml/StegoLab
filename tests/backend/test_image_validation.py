"""Exercise image source restrictions before expensive pixel allocation."""

import struct
import zlib
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from backend_service.failures import ApplicationFailure
from backend_service.image_preparation import inspect_image, read_prepared_png
from backend_service.image_validation import PNG_SIGNATURE, inspect_header, read_source
from schemas.images import MAXIMUM_IMAGE_FILE_BYTES, ImageSummary


def png_chunk(name: bytes, content: bytes) -> bytes:
    """Build an independently checksummed PNG chunk for malformed fixtures."""
    return (
        struct.pack(">I", len(content))
        + name
        + content
        + struct.pack(">I", zlib.crc32(name + content))
    )


def raw_png(width: int, height: int, depth: int = 8, color_type: int = 2) -> bytes:
    """Build a real PNG with independently encoded stored precision."""
    channels = 3 if color_type == 2 else 4
    row = b"\0" + b"\x23" * (width * channels * (depth // 8))
    return (
        PNG_SIGNATURE
        + png_chunk(
            b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0)
        )
        + png_chunk(b"IDAT", zlib.compress(row * height))
        + png_chunk(b"IEND", b"")
    )


def encoded(mode: str = "RGB", format_name: str = "PNG") -> bytes:
    """Write a small valid source in the requested image mode."""
    stream = BytesIO()
    Image.new(mode, (512, 512)).save(stream, format=format_name)
    return stream.getvalue()


@pytest.mark.parametrize("size", [(512, 512), (513, 515), (3840, 2160), (4096, 2160)])
def test_supported_dimensions_include_both_4k_sizes(size: tuple[int, int]) -> None:
    """Decode accepted odd, minimum, broadcast, and cinema dimensions."""
    result = inspect_image(BytesIO(raw_png(*size)))
    assert (result.source_width, result.source_height) == size


@pytest.mark.parametrize("size", [(511, 512), (512, 511), (4097, 512), (4096, 4096)])
def test_dimensions_fail_before_pillow_allocation(
    size: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject side or area limits from the header without decoding pixels."""
    data = raw_png(*size)

    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail if a rejected header is passed to the image decoder."""
        pytest.fail("Dimensions must be checked before Image.open.")

    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(ApplicationFailure) as failure:
        inspect_image(BytesIO(data))
    assert failure.value.code == "image_limits"


@pytest.mark.parametrize("color_type", [2, 6])
def test_real_sixteen_bit_png_rejected_before_downcast(
    color_type: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject real high-precision RGB/RGBA that Pillow would convert to eight bits."""
    source = raw_png(512, 512, depth=16, color_type=color_type)
    with Image.open(BytesIO(source)) as pillow_image:
        pillow_image.load()
        assert pillow_image.mode in ("RGB", "RGBA")

    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail if high-precision source reaches Pillow."""
        pytest.fail("Stored bit depth must be checked before Image.open.")

    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(ApplicationFailure):
        inspect_image(BytesIO(source))


@pytest.mark.parametrize(
    ("mode", "format_name"),
    [
        ("P", "PNG"),
        ("L", "PNG"),
        ("LA", "PNG"),
        ("CMYK", "JPEG"),
        ("L", "JPEG"),
        ("RGB", "TIFF"),
    ],
)
def test_unsupported_modes_and_formats(mode: str, format_name: str) -> None:
    """Require the selected eight-bit RGB and RGBA formats only."""
    with pytest.raises(ApplicationFailure):
        inspect_image(BytesIO(encoded(mode, format_name)))


def test_jpeg_is_accepted_but_not_as_recovery_input() -> None:
    """Support JPEG preparation while requiring exact PNG integers for recovery."""
    source = encoded(format_name="JPEG")
    assert inspect_image(BytesIO(source)).source_format == "JPEG"
    with pytest.raises(ApplicationFailure):
        read_prepared_png(BytesIO(source))


def test_twelve_bit_jpeg_header_is_rejected() -> None:
    """Read JPEG source precision explicitly before a decoder can alter it."""
    source = bytearray(encoded(format_name="JPEG"))
    index = source.index(b"\xff\xc0")
    source[index + 4] = 12
    with pytest.raises(ApplicationFailure):
        inspect_header(bytes(source))


def test_animation_and_color_key_transparency_are_rejected() -> None:
    """Do not silently discard frames or transparency from truecolor PNG."""
    image = Image.new("RGB", (512, 512), "red")
    animated = BytesIO()
    image.save(
        animated,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (512, 512), "blue")],
        duration=100,
    )
    transparent = BytesIO()
    image.save(transparent, format="PNG", transparency=(255, 0, 0))
    for source in (animated, transparent):
        with pytest.raises(ApplicationFailure):
            inspect_image(BytesIO(source.getvalue()))


@pytest.mark.parametrize(
    "damage", ["crc", "truncate", "invalid_pixels", "private_text"]
)
def test_malformed_input_fails_without_decoder_details(damage: str) -> None:
    """Cover damaged chunks, short streams, invalid pixels, and non-images."""
    source = encoded()
    if damage == "crc":
        source = source[:-1] + bytes([source[-1] ^ 1])
    elif damage == "truncate":
        source = source[: len(source) // 2]
    elif damage == "invalid_pixels":
        source = (
            source[:33]
            + png_chunk(b"IDAT", b"invalid-secret")
            + png_chunk(b"IEND", b"")
        )
    else:
        source = b"private-secret-nonimage"
    with pytest.raises(ApplicationFailure) as failure:
        inspect_image(BytesIO(source))
    assert "secret" not in str(failure.value)


def test_stream_and_file_limits_are_bounded(tmp_path: Path) -> None:
    """Stop at one excess byte for unknown-size streams and reject large files."""

    class EndlessStream(BytesIO):
        """Provide unlimited bytes without keeping a large fixture in memory."""

        count = 0

        def read(self, size: int | None = -1) -> bytes:
            """Record requested bytes and return only the bounded request."""
            assert size is not None and 0 < size <= 64 * 1024
            self.count += size
            return b"x" * size

    stream = EndlessStream()
    with pytest.raises(ApplicationFailure) as failure:
        read_source(stream)
    assert failure.value.code == "image_limits"
    assert stream.count == MAXIMUM_IMAGE_FILE_BYTES + 1
    oversized = tmp_path / "huge.png"
    with oversized.open("wb") as output:
        output.truncate(MAXIMUM_IMAGE_FILE_BYTES + 1)
    with pytest.raises(ApplicationFailure) as failure:
        inspect_image(oversized)
    assert failure.value.code == "image_limits"
    assert (
        len(read_source(BytesIO(b"x" * MAXIMUM_IMAGE_FILE_BYTES)))
        == MAXIMUM_IMAGE_FILE_BYTES
    )


def test_metadata_record_is_strict() -> None:
    """Reject dimension coercion, unknown fields, and invalid image area."""
    from pydantic import ValidationError

    values = inspect_image(BytesIO(encoded())).model_dump()
    for changes in (
        {"source_width": "512"},
        {"secret": "bad"},
        {"source_width": 4096, "source_height": 4096},
    ):
        with pytest.raises(ValidationError):
            ImageSummary.model_validate(values | changes)


def test_pillow_bomb_protection_remains_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Honor Pillow's allocation guard instead of disabling it during decoding."""
    source = encoded()
    assert Image.MAX_IMAGE_PIXELS is not None
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 200_000)
    with pytest.raises(ApplicationFailure):
        inspect_image(BytesIO(source))
    assert Image.MAX_IMAGE_PIXELS == 200_000
