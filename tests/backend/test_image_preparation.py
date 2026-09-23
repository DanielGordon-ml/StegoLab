"""Prove exact pixel preservation, orientation, and atomic output behavior."""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageCms, PngImagePlugin

from backend_service.failures import ApplicationFailure
from backend_service.image_preparation import (
    inspect_image,
    prepare_image,
    read_prepared_png,
)


def image_bytes(
    image: Image.Image, *, orientation: int = 1, color_profile: bytes | None = None
) -> bytes:
    """Encode an input fixture with orientation and optional RGB profile."""
    stream = BytesIO()
    metadata = Image.Exif()
    metadata[274] = orientation
    notes = PngImagePlugin.PngInfo()
    notes.add_text("private-source-note", "must-not-survive")
    image.save(
        stream, format="PNG", exif=metadata, pnginfo=notes, icc_profile=color_profile
    )
    return stream.getvalue()


def test_rgb_file_is_detected_by_content_and_metadata_removed(tmp_path: Path) -> None:
    """Use PNG bytes under a misleading suffix and retain every RGB value."""
    image = Image.new("RGB", (513, 515), (17, 61, 209))
    image.putpixel((400, 300), (1, 253, 144))
    source = tmp_path / "source.jpeg"
    source.write_bytes(image_bytes(image))
    destination = tmp_path / "prepared.png"
    summary = prepare_image(source, destination)
    assert summary == inspect_image(source)
    assert summary.source_format == "PNG"
    assert summary.color_policy == "assumed_srgb"
    recovered = read_prepared_png(destination)
    assert recovered.image.mode == "RGB"
    assert recovered.image.tobytes() == image.tobytes()
    with Image.open(destination) as result:
        assert result.info == {}
        assert len(result.getexif()) == 0
    assert "source.jpeg" not in summary.model_dump_json()


@pytest.mark.parametrize("orientation", range(1, 9))
def test_every_orientation_moves_alpha_and_color_once(
    tmp_path: Path, orientation: int
) -> None:
    """Compare marker coordinates against explicit orientation formulas."""
    width, height = 513, 515
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    markers = [(0, 0), (512, 0), (0, 514), (512, 514), (19, 71)]
    for index, point in enumerate(markers):
        image.putpixel(point, (index + 1, 20 + index, 33 + index, 41 + index))
    destination = tmp_path / "oriented.png"
    summary = prepare_image(
        BytesIO(image_bytes(image, orientation=orientation)), destination
    )
    recovered = read_prepared_png(destination)
    expected_size = (height, width) if orientation >= 5 else (width, height)
    assert recovered.image.size == expected_size
    assert (summary.prepared_width, summary.prepared_height) == expected_size
    for x, y in markers:
        positions = {
            1: (x, y),
            2: (width - 1 - x, y),
            3: (width - 1 - x, height - 1 - y),
            4: (x, height - 1 - y),
            5: (y, x),
            6: (height - 1 - y, x),
            7: (height - 1 - y, width - 1 - x),
            8: (y, width - 1 - x),
        }
        assert recovered.image.getpixel(positions[orientation]) == image.getpixel(
            (x, y)
        )
    assert recovered.image.histogram() == image.histogram()
    assert recovered.summary.orientation == 1


def test_color_conversion_uses_fixed_intent_and_preserves_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise a real RGB profile and inspect its fixed conversion options."""
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    original = Image.new("RGBA", (512, 513), (17, 39, 127, 40))
    original.putpixel((2, 3), (71, 42, 18, 211))
    convert = ImageCms.profileToProfile
    calls: list[tuple[object, object]] = []

    def tracked_conversion(*args: object, **kwargs: object) -> Image.Image:
        """Record the conversion policy and call the real color engine."""
        calls.append((kwargs["renderingIntent"], kwargs["flags"]))
        return convert(*args, **kwargs)  # type: ignore[arg-type, return-value]

    monkeypatch.setattr(ImageCms, "profileToProfile", tracked_conversion)
    output = tmp_path / "color.png"
    summary = prepare_image(
        BytesIO(image_bytes(original, color_profile=profile)), output
    )
    assert calls == [(ImageCms.Intent.RELATIVE_COLORIMETRIC, ImageCms.Flags(0))]
    assert summary.color_policy == "icc_to_srgb"
    assert read_prepared_png(output).image.tobytes() == original.tobytes()


def test_recovery_never_rotates_or_converts_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read tagged PNG integers without the source-preparation transforms."""
    image = Image.new("RGB", (512, 513), (20, 88, 140))
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail if preparation is accidentally used during recovery."""
        pytest.fail("Recovery must not transform pixels.")

    monkeypatch.setattr(ImageCms, "profileToProfile", forbidden)
    result = read_prepared_png(
        BytesIO(image_bytes(image, orientation=6, color_profile=profile))
    )
    assert result.image.size == image.size
    assert result.image.tobytes() == image.tobytes()
    assert result.image.info == {}
    assert result.summary.pixel_policy == "preserve_stored"


@pytest.mark.parametrize("existing", ["file", "symlink", "broken_symlink", "source"])
def test_preparation_never_replaces_existing_paths(
    tmp_path: Path, existing: str
) -> None:
    """Protect destination files, destination symlinks, and the source itself."""
    source = tmp_path / "source.png"
    source_bytes = image_bytes(Image.new("RGB", (512, 512)))
    source.write_bytes(source_bytes)
    output = tmp_path / "output.png"
    if existing == "file":
        output.write_bytes(b"existing-private-output")
    elif existing == "symlink":
        output.symlink_to(source)
    elif existing == "broken_symlink":
        output.symlink_to(tmp_path / "missing.png")
    else:
        output = source
    before = output.read_bytes() if existing != "broken_symlink" else None
    with pytest.raises(ApplicationFailure) as failure:
        prepare_image(source, output)
    assert failure.value.code == "image_write"
    if before is not None:
        assert output.read_bytes() == before
    else:
        assert output.is_symlink() and not output.exists()
    assert source.read_bytes() == source_bytes
    assert not list(tmp_path.glob(".stegolab-image-*"))


def test_publish_race_preserves_the_winning_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Create another output just before publication and refuse to replace it."""
    import os

    output = tmp_path / "output.png"
    link = os.link

    def competing_publish(source: Path, destination: Path) -> None:
        """Simulate another process winning after temporary-file verification."""
        destination.write_bytes(b"race-winner")
        link(source, destination)

    monkeypatch.setattr("backend_service.image_output.os.link", competing_publish)
    with pytest.raises(ApplicationFailure):
        prepare_image(BytesIO(image_bytes(Image.new("RGB", (512, 512)))), output)
    assert output.read_bytes() == b"race-winner"
    assert not list(tmp_path.glob(".stegolab-image-*"))


@pytest.mark.parametrize("operation", ["save", "fsync", "verify"])
def test_failed_write_is_clean_and_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Remove incomplete output and suppress private operating-system details."""
    source = tmp_path / "private-source.png"
    source.write_bytes(image_bytes(Image.new("RGBA", (512, 512))))
    original = source.read_bytes()
    output = tmp_path / "private-output.png"

    def fail(*args: object, **kwargs: object) -> None:
        """Model a low-level write failure containing sensitive details."""
        raise OSError("private-source.png secret-path sentinel")

    if operation == "save":
        monkeypatch.setattr(Image.Image, "save", fail)
    elif operation == "fsync":
        monkeypatch.setattr("backend_service.image_output.os.fsync", fail)
    else:
        monkeypatch.setattr("backend_service.image_output.ImageChops.difference", fail)
    with pytest.raises(ApplicationFailure) as failure:
        prepare_image(source, output)
    assert failure.value.code == "image_write"
    assert "private" not in str(failure.value) and "sentinel" not in str(failure.value)
    assert source.read_bytes() == original
    assert not output.exists()
    assert not list(tmp_path.glob(".stegolab-image-*"))


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_pixel_verification_prevents_corrupted_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Detect a changed color byte even when the corresponding alpha is zero."""
    source = BytesIO(image_bytes(Image.new(mode, (512, 512))))
    original_save = Image.Image.save

    def corrupt_save(image: Image.Image, *args: object, **kwargs: object) -> None:
        """Write different pixels while leaving format and dimensions valid."""
        changed = image.copy()
        changed.putpixel((0, 0), (1, 0, 0, 0) if mode == "RGBA" else (1, 0, 0))
        original_save(changed, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Image.Image, "save", corrupt_save)
    output = tmp_path / "output.png"
    with pytest.raises(ApplicationFailure) as failure:
        prepare_image(source, output)
    assert failure.value.code == "image_write"
    assert not output.exists()
    assert not list(tmp_path.glob(".stegolab-image-*"))
