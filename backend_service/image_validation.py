"""Read bounded input and check stored image headers before decoding."""

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from backend_service.failures import ApplicationFailure
from backend_service.image_png_stream import PixelStreamValidator
from backend_service.image_policy import PRODUCTION_IMAGE_POLICY, ImagePolicy

ImageSource = Path | BinaryIO
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class SourceHeader:
    """Keep validated stored dimensions and format before pixel allocation."""

    format: Literal["JPEG", "PNG"]
    width: int
    height: int
    mode: Literal["L", "RGB", "RGBA"]
    png_chunks: frozenset[bytes] = frozenset()
    has_profile: bool = False


def image_failure(code: str = "image_invalid") -> ApplicationFailure:
    """Return a fixed failure with no input bytes, paths, or decoder details."""
    messages = {
        "image_invalid": "The image is invalid or unsupported. Convert it to a "
        "single-frame, eight-bit RGB JPEG or RGB/RGBA PNG, then retry.",
        "image_limits": "The image exceeds supported limits. Use sides of "
        "512–4096 pixels, at most 8,850,000 pixels, and at most 50 MiB.",
        "image_color": "The image color declaration is invalid or unsupported. "
        "Convert the image to eight-bit sRGB with a valid RGB profile, then retry.",
        "image_resources": "There is not enough memory to process the image. "
        "Close other applications or use a smaller image, then retry.",
        "image_write": "The prepared image could not be saved. Choose a new "
        "output file and check directory access and free space, then retry.",
    }
    status_code = 503 if code == "image_resources" else 422
    return ApplicationFailure(code, messages[code], status_code=status_code)


def _read_stream(stream: BinaryIO, policy: ImagePolicy) -> bytes:
    """Limit reads even for a stream with no known file size."""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = stream.read(min(64 * 1024, policy.maximum_bytes + 1 - size))
        if not isinstance(chunk, bytes):
            raise image_failure()
        size += len(chunk)
        if size > policy.maximum_bytes:
            raise policy.limit_failure()
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def read_source(
    source: ImageSource, policy: ImagePolicy = PRODUCTION_IMAGE_POLICY
) -> bytes:
    """Read from a path or the current stream position within the byte limit."""
    try:
        if isinstance(source, Path):
            if source.stat().st_size > policy.maximum_bytes:
                raise policy.limit_failure()
            with source.open("rb") as stream:
                return _read_stream(stream, policy)
        return _read_stream(source, policy)
    except ApplicationFailure:
        raise
    except (OSError, ValueError, TypeError, AttributeError):
        raise image_failure() from None


def check_dimensions(
    width: int, height: int, policy: ImagePolicy = PRODUCTION_IMAGE_POLICY
) -> None:
    """Apply trusted bounds before allocating decoded pixels."""
    policy.check_dimensions(width, height)


def _png_header(
    data: bytes, policy: ImagePolicy, validate_pixels: bool
) -> SourceHeader:
    """Check source bit depth, color declarations, chunks, and dimensions."""
    offset, first = 8, True
    seen: set[bytes] = set()
    width = height = color_type = 0
    saw_end = False
    pixels_finished = False
    pixel_stream: PixelStreamValidator | None = None
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        name = data[offset + 4 : offset + 8]
        end = offset + length + 12
        if end > len(data):
            raise image_failure()
        content = data[offset + 8 : end - 4]
        checksum = int.from_bytes(data[end - 4 : end], "big")
        if (
            len(name) != 4
            or not all(65 <= value <= 90 or 97 <= value <= 122 for value in name)
            or not 65 <= name[2] <= 90
            or (
                65 <= name[0] <= 90 and name not in {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
            )
        ):
            raise image_failure()
        if zlib.crc32(name + content) != checksum:
            raise image_failure()
        if first:
            if name != b"IHDR" or length != 13:
                raise image_failure()
            width, height, depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", content)
            )
            check_dimensions(width, height, policy)
            if depth != 8 or color_type not in (
                (0, 2, 6) if policy.allow_grayscale else (2, 6)
            ):
                raise image_failure()
            if compression != 0 or filtering != 0 or interlace not in (0, 1):
                raise image_failure()
            if validate_pixels:
                channels = {0: 1, 2: 3, 6: 4}[color_type]
                pixel_stream = PixelStreamValidator(width, height, channels, interlace)
            first = False
        elif name == b"IHDR":
            raise image_failure()
        if name == b"PLTE" and (
            name in seen or b"IDAT" in seen or not 0 < length <= 768 or length % 3
        ):
            raise image_failure()
        if name == b"IDAT":
            if pixels_finished:
                raise image_failure()
            if pixel_stream is not None:
                pixel_stream.feed(content)
        elif b"IDAT" in seen:
            pixels_finished = True
        if name in (b"acTL", b"fcTL", b"fdAT", b"tRNS"):
            raise image_failure()
        if name == b"cICP":
            raise image_failure("image_color")
        if name in (b"iCCP", b"sRGB", b"gAMA", b"cHRM"):
            if name in seen or b"IDAT" in seen:
                raise image_failure("image_color")
            if name == b"sRGB" and (length != 1 or content[0] > 3):
                raise image_failure("image_color")
            if name == b"gAMA" and (length != 4 or content == b"\0" * 4):
                raise image_failure("image_color")
            if name == b"cHRM" and length != 32:
                raise image_failure("image_color")
            if name == b"iCCP":
                _check_profile_chunk(content)
        seen.add(name)
        offset = end
        if name == b"IEND":
            if length != 0 or offset != len(data):
                raise image_failure()
            saw_end = True
            break
    if first or not saw_end or b"IDAT" not in seen:
        raise image_failure()
    if pixel_stream is not None:
        pixel_stream.finish()
    if seen & {b"gAMA", b"cHRM"} and not seen & {b"iCCP", b"sRGB"}:
        raise image_failure("image_color")
    return SourceHeader(
        "PNG",
        width,
        height,
        "L" if color_type == 0 else "RGBA" if color_type == 6 else "RGB",
        frozenset(seen),
    )


def _check_profile_chunk(content: bytes) -> None:
    """Reject malformed PNG profile names or compression methods early."""
    name, separator, payload = content.partition(b"\0")
    if (
        not separator
        or not 1 <= len(name) <= 79
        or any(value < 32 or 126 < value < 161 for value in name)
        or name.startswith(b" ")
        or name.endswith(b" ")
        or b"  " in name
        or len(payload) < 2
        or payload[0] != 0
    ):
        raise image_failure("image_color")
    try:
        decoder = zlib.decompressobj()
        profile = decoder.decompress(payload[1:], 1024 * 1024 + 1)
        if len(profile) > 1024 * 1024 or not decoder.eof or decoder.unused_data:
            raise image_failure("image_color")
    except zlib.error:
        raise image_failure("image_color") from None


def _jpeg_header(data: bytes, policy: ImagePolicy) -> SourceHeader:
    """Read source precision and RGB component count before Pillow decoding."""
    offset = 2
    frame: SourceHeader | None = None
    profiles: dict[int, int] = {}
    while offset < len(data):
        if data[offset] != 0xFF:
            raise image_failure()
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xD9, 0xDA):
            break
        if marker in (0x01, *range(0xD0, 0xD9)):
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset : offset + 2], "big")
        end = offset + length
        if length < 2 or end > len(data):
            raise image_failure()
        if marker == 0xE2 and data[offset + 2 : offset + 14] == b"ICC_PROFILE\0":
            if length < 16:
                raise image_failure("image_color")
            sequence, total = data[offset + 14 : offset + 16]
            if not 1 <= sequence <= total or sequence in profiles:
                raise image_failure("image_color")
            profiles[sequence] = total
        if marker in set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}:
            if marker not in (0xC0, 0xC1, 0xC2) or length < 8:
                raise image_failure()
            precision, height, width, components = struct.unpack(
                ">BHHB", data[offset + 2 : offset + 8]
            )
            check_dimensions(width, height, policy)
            if (
                precision != 8
                or components not in ((1, 3) if policy.allow_grayscale else (3,))
                or length != 8 + components * 3
            ):
                raise image_failure()
            if frame is not None:
                raise image_failure()
            frame = SourceHeader(
                "JPEG", width, height, "L" if components == 1 else "RGB"
            )
        offset = end
    if frame is None:
        raise image_failure()
    if profiles and (
        set(profiles.values()) != {len(profiles)}
        or set(profiles) != set(range(1, len(profiles) + 1))
    ):
        raise image_failure("image_color")
    return SourceHeader(
        "JPEG", frame.width, frame.height, frame.mode, has_profile=bool(profiles)
    )


def inspect_header(
    data: bytes,
    policy: ImagePolicy = PRODUCTION_IMAGE_POLICY,
    *,
    validate_pixels: bool = True,
) -> SourceHeader:
    """Select supported formats from file bytes rather than the extension."""
    if len(data) > policy.maximum_bytes:
        raise policy.limit_failure()
    if data.startswith(PNG_SIGNATURE):
        try:
            return _png_header(data, policy, validate_pixels)
        except (ValueError, zlib.error):
            raise image_failure() from None
    if data.startswith(b"\xff\xd8"):
        return _jpeg_header(data, policy)
    raise image_failure()
