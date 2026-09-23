"""Check declared color precedence and safe rejection of invalid metadata."""

import struct
import warnings
import zlib
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image, ImageCms

from backend_service.failures import ApplicationFailure
from backend_service.image_preparation import (
    inspect_image,
    prepare_image,
    read_prepared_png,
)


def color_png(chunks: list[tuple[bytes, bytes]]) -> bytes:
    """Insert independently encoded color chunks before a valid PNG's pixels."""
    stream = BytesIO()
    Image.new("RGBA", (512, 512), (78, 64, 170, 91)).save(stream, format="PNG")
    additions = b""
    for name, content in chunks:
        additions += (
            struct.pack(">I", len(content))
            + name
            + content
            + struct.pack(">I", zlib.crc32(name + content))
        )
    source = stream.getvalue()
    return source[:33] + additions + source[33:]


def profile_chunk(color_space: Literal["LAB", "XYZ", "sRGB"] = "sRGB") -> bytes:
    """Provide a complete color profile in a PNG declaration."""
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile(color_space)).tobytes()
    return b"Profile\0\0" + zlib.compress(profile)


@pytest.mark.parametrize(
    ("chunks", "policy"),
    [
        ([], "assumed_srgb"),
        ([(b"sRGB", b"\0")], "declared_srgb"),
        ([(b"sRGB", b"\0"), (b"gAMA", struct.pack(">I", 45455))], "declared_srgb"),
        ([(b"iCCP", profile_chunk()), (b"sRGB", b"\0")], "icc_to_srgb"),
    ],
)
def test_color_policy_precedence(
    chunks: list[tuple[bytes, bytes]], policy: str
) -> None:
    """Choose valid RGB profiles before sRGB tags and otherwise state assumptions."""
    assert inspect_image(BytesIO(color_png(chunks))).color_policy == policy


@pytest.mark.parametrize(
    "chunks",
    [
        [(b"gAMA", struct.pack(">I", 45455))],
        [(b"cHRM", b"\0" * 32)],
        [(b"cICP", bytes([1, 13, 0, 1]))],
        [(b"sRGB", b"\4")],
        [(b"sRGB", b"")],
        [(b"sRGB", b"\0\0")],
        [(b"sRGB", b"\0"), (b"sRGB", b"\0")],
        [(b"iCCP", b"Profile\0\0" + zlib.compress(b"secret-invalid-profile"))],
        [(b"iCCP", b"Profile\0\0bad-compressed-secret")],
        [(b"iCCP", b"Profile\0\1unsupported-compression")],
        [(b"iCCP", b"\0\0" + zlib.compress(b"invalid"))],
        [(b"iCCP", profile_chunk("LAB"))],
        [(b"sRGB", b"\0"), (b"gAMA", b"\0" * 4)],
        [(b"sRGB", b"\0"), (b"cHRM", b"too-short")],
    ],
)
def test_unsupported_and_malformed_declarations(
    chunks: list[tuple[bytes, bytes]],
) -> None:
    """Reject ambiguous or damaged color declarations with conversion guidance."""
    with pytest.raises(ApplicationFailure) as failure:
        inspect_image(BytesIO(color_png(chunks)))
    assert failure.value.code == "image_color"
    assert "Convert" in str(failure.value)
    assert "secret" not in str(failure.value)


def test_decompressed_profile_limit_is_bounded() -> None:
    """Reject a profile expansion before handing it to Pillow's color engine."""
    chunk = b"Profile\0\0" + zlib.compress(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ApplicationFailure) as failure:
        inspect_image(BytesIO(color_png([(b"iCCP", chunk)])))
    assert failure.value.code == "image_color"


def test_jpeg_profile_is_validated_and_not_silently_dropped(tmp_path: Path) -> None:
    """Honor a real JPEG RGB profile and reject incomplete profile segments."""
    stream = BytesIO()
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (513, 512), (19, 77, 83)).save(
        stream, format="JPEG", icc_profile=profile
    )
    source = stream.getvalue()
    summary = prepare_image(BytesIO(source), tmp_path / "prepared.png")
    assert summary.color_policy == "icc_to_srgb"
    assert read_prepared_png(tmp_path / "prepared.png").image.mode == "RGB"
    damaged = bytearray(source)
    sequence_position = damaged.index(b"ICC_PROFILE\0") + len(b"ICC_PROFILE\0")
    damaged[sequence_position + 1] = 2
    with pytest.raises(ApplicationFailure) as failure:
        inspect_image(BytesIO(damaged))
    assert failure.value.code == "image_color"


def test_metadata_warnings_are_safe_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent decoder warnings from writing source metadata to logs or stderr."""
    from backend_service import image_preparation

    def warning(*args: object, **kwargs: object) -> None:
        """Simulate a decoder warning with private metadata details."""
        warnings.warn("secret-private-metadata", UserWarning, stacklevel=1)

    monkeypatch.setattr(image_preparation, "resolve_color", warning)
    with pytest.raises(ApplicationFailure) as failure:
        inspect_image(BytesIO(color_png([])))
    assert "secret" not in str(failure.value)
