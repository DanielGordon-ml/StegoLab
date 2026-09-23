"""Prepare bounded dataset images without changing production image limits."""

import hashlib
import os
import struct
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

from backend_service.dataset_files import (
    SourceFile,
    directory_descriptor,
    read_source_bytes,
    stat_identity,
)
from backend_service.failures import ApplicationFailure
from backend_service.image_color import ColorPolicy, convert_color, resolve_color
from backend_service.image_diagnostics import private_image_diagnostics
from backend_service.image_output import write_png
from backend_service.image_policy import DATASET_PREPARED_POLICY, DATASET_SOURCE_POLICY
from backend_service.image_preparation import _open_pixels
from backend_service.image_validation import (
    SourceHeader,
    image_failure,
    inspect_header,
)

RGB_DIGEST_DOMAIN = b"StegoLab/rgb-pixels/v1\x00"


@dataclass(frozen=True)
class DatasetPreparedImage:
    """Describe prepared pixels and checksums without source paths or metadata."""

    source_format: Literal["JPEG", "PNG"]
    source_mode: Literal["L", "RGB", "RGBA"]
    source_width: int
    source_height: int
    width: int
    height: int
    mode: Literal["RGB", "RGBA"]
    orientation: int
    color_policy: ColorPolicy
    grayscale_converted: bool
    eligible: bool
    prepared_checksum: str
    rgb_checksum: str
    prepared_bytes: int


@contextmanager
def _safe_image_operation() -> Iterator[None]:
    """Hide parser details while retaining safe application failures."""
    try:
        with private_image_diagnostics(), warnings.catch_warnings():
            warnings.simplefilter("error")
            yield
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
        struct.error,
        Image.DecompressionBombError,
        Warning,
    ):
        raise image_failure() from None


def inspect_dataset_header(data: bytes) -> SourceHeader:
    """Inspect bounded stored headers without decoding PNG or JPEG pixels."""
    with _safe_image_operation():
        return inspect_header(data, DATASET_SOURCE_POLICY, validate_pixels=False)


def rgb_pixel_checksum(image: Image.Image) -> str:
    """Hash the domain, big-endian dimensions, and row-major RGB bytes."""
    digest = hashlib.sha256(RGB_DIGEST_DOMAIN)
    digest.update(struct.pack(">II", image.width, image.height))
    for top in range(0, image.height, 64):
        with image.crop((0, top, image.width, min(top + 64, image.height))) as strip:
            with strip.convert("RGB") as rgb:
                digest.update(rgb.tobytes())
    return digest.hexdigest()


def _file_checksum(path: Path) -> tuple[str, int]:
    """Hash bounded prepared bytes after verified publication."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            size += len(chunk)
            if size > DATASET_PREPARED_POLICY.maximum_bytes:
                raise DATASET_PREPARED_POLICY.limit_failure()
            digest.update(chunk)
    return digest.hexdigest(), size


def _describe(
    image: Image.Image,
    header: SourceHeader,
    orientation: int,
    color_policy: ColorPolicy,
    checksum: str,
    size: int,
) -> DatasetPreparedImage:
    """Build source provenance and the exact prepared-pixel description."""
    if image.mode not in ("RGB", "RGBA"):
        raise image_failure()
    return DatasetPreparedImage(
        source_format=header.format,
        source_mode=header.mode,
        source_width=header.width,
        source_height=header.height,
        width=image.width,
        height=image.height,
        mode="RGBA" if image.mode == "RGBA" else "RGB",
        orientation=orientation,
        color_policy=color_policy,
        grayscale_converted=header.mode == "L",
        eligible=min(image.size) >= 256,
        prepared_checksum=checksum,
        rgb_checksum=rgb_pixel_checksum(image),
        prepared_bytes=size,
    )


def prepare_dataset_image(
    data: bytes, destination: Path, before_write: Callable[[int], None]
) -> DatasetPreparedImage:
    """Prepare one source, reserve bounded writes, and verify its saved PNG."""
    with _safe_image_operation():
        header = inspect_header(data, DATASET_SOURCE_POLICY)
        clean, orientation, color_policy = _clean_pixels(data, header)
        with clean:
            write_png(clean, destination, before_write)
            checksum, size = _file_checksum(destination)
            return _describe(clean, header, orientation, color_policy, checksum, size)


def _clean_pixels(
    data: bytes, header: SourceHeader
) -> tuple[Image.Image, int, ColorPolicy]:
    """Release source and conversion buffers before saving and reopening output."""
    with _open_pixels(data, header) as original:
        orientation = original.getexif().get(274, 1)
        if type(orientation) is not int or not 1 <= orientation <= 8:
            raise image_failure()
        color_policy, profile = resolve_color(original, header)
        if header.mode == "L" and profile is not None:
            raise image_failure("image_color")
        with ImageOps.exif_transpose(original) as oriented:
            converted = (
                oriented.convert("RGB")
                if header.mode == "L"
                else convert_color(oriented, profile)
            )
            with converted:
                clean = Image.new(converted.mode, converted.size)
                try:
                    clean.paste(converted)
                except BaseException:
                    clean.close()
                    raise
                return clean, orientation, color_policy


def validate_dataset_png(path: Path) -> DatasetPreparedImage:
    """Read prepared PNG pixels unchanged and report their integrity fields."""
    with _safe_image_operation():
        data = _read_prepared_bytes(path)
        header = inspect_header(data, DATASET_PREPARED_POLICY)
        if header.format != "PNG":
            raise image_failure()
        with _open_pixels(data, header) as prepared:
            color_policy, _ = resolve_color(prepared, header)
            return _describe(
                prepared,
                header,
                1,
                color_policy,
                hashlib.sha256(data).hexdigest(),
                len(data),
            )


def _read_prepared_bytes(path: Path) -> bytes:
    """Anchor every path component and reject file changes during validation."""
    try:
        with directory_descriptor(path.parent) as parent:
            information = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if information.st_size > DATASET_PREPARED_POLICY.maximum_bytes:
            raise DATASET_PREPARED_POLICY.limit_failure()
        source = SourceFile(
            path.name, information.st_size, stat_identity(information), "image"
        )
        return read_source_bytes(
            path.parent, source, DATASET_PREPARED_POLICY.maximum_bytes
        )
    except ApplicationFailure as failure:
        if failure.code.startswith("dataset_"):
            raise image_failure() from None
        raise
