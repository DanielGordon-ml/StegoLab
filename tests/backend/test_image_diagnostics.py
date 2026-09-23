"""Prove source metadata cannot escape through enabled Pillow debug logs."""

import logging
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageCms

from backend_service.image_diagnostics import private_image_diagnostics
from backend_service.image_preparation import (
    inspect_image,
    prepare_image,
    read_prepared_png,
)


def private_metadata_source() -> bytes:
    """Create valid pixels carrying private ICC-name and EXIF sentinel values."""
    metadata = Image.Exif()
    metadata[0x010E] = "private-exif-sentinel"
    stream = BytesIO()
    Image.new("RGB", (512, 512), (50, 65, 88)).save(stream, format="PNG", exif=metadata)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    payload = b"private-profile-sentinel\0\0" + zlib.compress(profile)
    profile_chunk = (
        struct.pack(">I", len(payload))
        + b"iCCP"
        + payload
        + struct.pack(">I", zlib.crc32(b"iCCP" + payload))
    )
    source = stream.getvalue()
    return source[:33] + profile_chunk + source[33:]


@pytest.mark.parametrize("operation", ["inspect", "prepare", "recover"])
def test_source_metadata_never_reaches_debug_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, operation: str
) -> None:
    """Suppress concrete metadata loggers during every public image operation."""
    source = private_metadata_source()
    caplog.clear()
    caplog.set_level(logging.DEBUG)
    if operation == "inspect":
        inspect_image(BytesIO(source))
    elif operation == "prepare":
        prepare_image(BytesIO(source), tmp_path / "prepared.png")
    else:
        read_prepared_png(BytesIO(source))
    assert "private-exif-sentinel" not in caplog.text
    assert "private-profile-sentinel" not in caplog.text
    logger = logging.getLogger("PIL.PngImagePlugin")
    logger.debug("outside-image-service-diagnostic")
    assert "outside-image-service-diagnostic" in caplog.text


def test_write_logs_are_private_and_other_threads_remain_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Scope suppression to service context instead of changing global levels."""
    source = private_metadata_source()
    save = Image.Image.save
    logger = logging.getLogger("PIL.PngImagePlugin")
    caplog.clear()
    caplog.set_level(logging.DEBUG)

    def logging_save(image: Image.Image, *args: object, **kwargs: object) -> None:
        """Expose a simulated write diagnostic to the same Pillow logger."""
        logger.debug("private-write-sentinel")
        save(image, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Image.Image, "save", logging_save)
    prepare_image(BytesIO(source), tmp_path / "prepared.png")
    assert "private-write-sentinel" not in caplog.text
    with private_image_diagnostics(), ThreadPoolExecutor(max_workers=1) as workers:
        logger.debug("private-current-context-sentinel")
        workers.submit(logger.debug, "unrelated-thread-diagnostic").result()
    assert "private-current-context-sentinel" not in caplog.text
    assert "unrelated-thread-diagnostic" in caplog.text
    logger.debug("outside-after-context-diagnostic")
    assert "outside-after-context-diagnostic" in caplog.text
