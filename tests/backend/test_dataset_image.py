"""Check dataset pixel preparation without widening production image support."""

import hashlib
import struct
import zlib
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from backend_service.dataset_image import (
    inspect_dataset_header,
    prepare_dataset_image,
    rgb_pixel_checksum,
    validate_dataset_png,
)
from backend_service.failures import ApplicationFailure
from backend_service.image_policy import DATASET_PREPARED_POLICY, DATASET_SOURCE_POLICY
from backend_service.image_preparation import inspect_image


def encoded_image(
    mode: str = "RGB",
    size: tuple[int, int] = (256, 257),
    format_name: str = "PNG",
    orientation: int = 1,
) -> bytes:
    """Build a patterned public fixture with explicit orientation."""
    output = BytesIO()
    with Image.new(mode, size) as image:
        image.paste(91 if mode == "L" else (31, 79, 127), (0, 0, size[0] // 2, size[1]))
        exif = Image.Exif()
        exif[274] = orientation
        image.save(output, format=format_name, exif=exif)
    return output.getvalue()


@pytest.mark.parametrize("format_name", ["JPEG", "PNG"])
def test_grayscale_is_expanded_exactly_once(tmp_path: Path, format_name: str) -> None:
    """Convert oriented grayscale to equal RGB channels and strip metadata."""
    source = encoded_image("L", format_name=format_name, orientation=6)
    destination = tmp_path / "prepared.png"
    writes: list[int] = []
    result = prepare_dataset_image(source, destination, writes.append)
    assert result.source_mode == "L"
    assert result.grayscale_converted
    assert (result.width, result.height) == (257, 256)
    assert result.eligible
    assert result.orientation == 6
    assert result.color_policy == "assumed_srgb"
    assert sum(writes) == result.prepared_bytes == destination.stat().st_size
    assert len(writes) > 1
    with Image.open(BytesIO(source)) as original:
        expected = original.transpose(Image.Transpose.ROTATE_270).convert("RGB")
        with expected, Image.open(destination) as reopened:
            assert reopened.mode == "RGB"
            assert reopened.info == {}
            assert reopened.tobytes() == expected.tobytes()
            red, green, blue = reopened.split()
            with red, green, blue:
                assert red.tobytes() == green.tobytes() == blue.tobytes()
    validated = validate_dataset_png(destination)
    assert validated.prepared_checksum == result.prepared_checksum
    assert validated.rgb_checksum == result.rgb_checksum
    assert (validated.width, validated.height) == (result.width, result.height)
    with pytest.raises(ApplicationFailure):
        inspect_image(BytesIO(source))


@pytest.mark.parametrize("size", [(1, 1), (255, 256), (256, 256), (3840, 2560)])
def test_dataset_sizes_and_crop_eligibility(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """Accept dataset sizes independently and report the 256-pixel crop rule."""
    source = encoded_image(size=size)
    result = prepare_dataset_image(source, tmp_path / "prepared.png", lambda size: None)
    assert (result.width, result.height) == size
    assert result.eligible == (min(size) >= 256)
    with pytest.raises(ApplicationFailure):
        inspect_image(BytesIO(source))


def test_rgb_digest_excludes_alpha_and_includes_dimensions(tmp_path: Path) -> None:
    """Give identical RGB covers one digest despite alpha differences."""
    outputs = []
    for alpha in (0, 255):
        source = BytesIO()
        with Image.new("RGBA", (256, 257), (11, 31, 79, alpha)) as image:
            image.save(source, format="PNG")
        destination = tmp_path / f"prepared-{alpha}.png"
        result = prepare_dataset_image(
            source.getvalue(), destination, lambda size: None
        )
        outputs.append(result)
        with Image.open(destination) as reopened:
            assert reopened.getpixel((0, 0)) == (11, 31, 79, alpha)
    assert outputs[0].rgb_checksum == outputs[1].rgb_checksum
    assert outputs[0].prepared_checksum != outputs[1].prepared_checksum
    expected = hashlib.sha256(
        b"StegoLab/rgb-pixels/v1\x00"
        + struct.pack(">II", 256, 257)
        + bytes((11, 31, 79)) * (256 * 257)
    ).hexdigest()
    assert outputs[0].rgb_checksum == expected
    with Image.new("RGB", (257, 256), (11, 31, 79)) as swapped:
        assert rgb_pixel_checksum(swapped) != expected


def test_header_inventory_does_not_decode_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid both Pillow loading and PNG pixel decompression during inventory."""
    sources = [
        encoded_image(format_name=format_name) for format_name in ("JPEG", "PNG")
    ]

    def forbidden(*arguments: object, **keywords: object) -> None:
        """Fail if an inventory call tries to allocate or inflate pixels."""
        pytest.fail("Header inventory must not decode pixels.")

    monkeypatch.setattr(Image, "open", forbidden)
    monkeypatch.setattr(
        "backend_service.image_png_stream.PixelStreamValidator.feed", forbidden
    )
    for source in sources:
        header = inspect_dataset_header(source)
        assert (header.width, header.height, header.mode) == (256, 257, "RGB")


def test_write_budget_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    """Stop before a rejected chunk and remove the owned temporary file."""
    destination = tmp_path / "prepared.png"

    def refuse_write(size: int) -> None:
        """Model the dataset root's insufficient-space response."""
        raise ApplicationFailure("dataset_space", "Insufficient free space.")

    with pytest.raises(ApplicationFailure, match="Insufficient free space"):
        prepare_dataset_image(encoded_image(), destination, refuse_write)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_prepared_file_limit_is_enforced_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Count actual encoded bytes and stop safely at the prepared-file ceiling."""
    monkeypatch.setattr(
        "backend_service.image_output.DATASET_PREPARED_POLICY",
        replace(DATASET_PREPARED_POLICY, maximum_bytes=32),
    )
    writes: list[int] = []
    with pytest.raises(ApplicationFailure) as failure:
        prepare_dataset_image(encoded_image(), tmp_path / "prepared.png", writes.append)
    assert failure.value.code == "image_limits"
    assert sum(writes) <= 32
    assert list(tmp_path.iterdir()) == []


def test_existing_destination_and_invalid_formats_fail(tmp_path: Path) -> None:
    """Protect previous output and reject unsupported dataset pixel modes."""
    destination = tmp_path / "prepared.png"
    destination.write_bytes(b"previous-output")
    with pytest.raises(ApplicationFailure):
        prepare_dataset_image(encoded_image(), destination, lambda size: None)
    assert destination.read_bytes() == b"previous-output"
    for mode in ("P", "LA"):
        source = BytesIO()
        with Image.new(mode, (256, 256)) as image:
            image.save(source, format="PNG")
        with pytest.raises(ApplicationFailure):
            inspect_dataset_header(source.getvalue())


@pytest.mark.parametrize("size", [(0, 256), (8193, 256), (8192, 3907)])
def test_dataset_dimensions_fail_before_allocation(
    size: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject invalid side and area bounds from stored dimensions alone."""
    source = encoded_image()
    content = struct.pack(">II", *size) + source[24:29]
    source = (
        source[:16]
        + content
        + struct.pack(">I", zlib.crc32(b"IHDR" + content))
        + source[33:]
    )

    def forbidden(*arguments: object, **keywords: object) -> None:
        """Prevent decoding any image whose stored dimensions are invalid."""
        pytest.fail("Invalid dimensions must not reach the decoder.")

    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(ApplicationFailure) as failure:
        inspect_dataset_header(source)
    assert failure.value.code == "image_limits"
    assert "1–8192" in failure.value.message


def test_prepared_reader_has_a_separate_byte_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep expanded prepared PNGs readable under their distinct file limit."""
    source = encoded_image()
    destination = tmp_path / "prepared.png"
    result = prepare_dataset_image(source, destination, lambda size: None)
    monkeypatch.setattr(
        "backend_service.dataset_image.DATASET_SOURCE_POLICY",
        replace(DATASET_SOURCE_POLICY, maximum_bytes=32),
    )
    with pytest.raises(ApplicationFailure):
        inspect_dataset_header(source)
    assert (
        validate_dataset_png(destination).prepared_checksum == result.prepared_checksum
    )


def test_invalid_png_pixels_are_not_claimed_valid_by_header_inventory(
    tmp_path: Path,
) -> None:
    """Separate header acceptance from the full preparation integrity check."""
    source = encoded_image()
    content = b"invalid-compressed-pixels"
    source = (
        source[:33]
        + struct.pack(">I", len(content))
        + b"IDAT"
        + content
        + struct.pack(">I", zlib.crc32(b"IDAT" + content))
        + source[-12:]
    )
    assert inspect_dataset_header(source).mode == "RGB"
    with pytest.raises(ApplicationFailure):
        prepare_dataset_image(source, tmp_path / "prepared.png", lambda size: None)
    assert list(tmp_path.iterdir()) == []
