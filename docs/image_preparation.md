# Image preparation

Sprint 2 prepares image pixels for later model work. It does not hide text or
claim that an image can survive a model, resizing, screenshots, or JPEG saving.

## Commands

From the project root:

```sh
uv run stegolab inspect_image /path/to/source.jpg
uv run stegolab prepare_image /path/to/source.jpg /path/to/prepared.png
```

- `inspect_image` validates the full source and reports safe metadata. It does
  the preparation in memory so unsupported color conversions fail at inspection.
- `prepare_image` writes a new PNG and reports the same source/preparation
  metadata. An existing output, including a symlink, causes a safe failure.
- Commands accept image paths only. No passwords or message text are needed.
- Metadata contains no paths, source comments, profiles, or private values.

## Accepted sources

- Single-frame, eight-bit RGB JPEG or RGB/RGBA PNG, detected from file bytes.
- Each side: 512–4096 pixels. Total area: at most 8,850,000 pixels.
- Both 3840 × 2160 and 4096 × 2160 are supported. Odd dimensions are supported.
- Input size: at most 50 MiB (52,428,800 bytes), including unknown-size streams.
- Source headers are checked before decoded pixel allocation. In particular,
  sixteen-bit RGB/RGBA PNG is rejected before Pillow can reduce its precision.
- Grayscale, palette, CMYK, animation, and RGB PNG color-key transparency are
  rejected. Convert these to single-frame eight-bit sRGB RGB/RGBA first.
- PNG validation checks chunk order and the complete compressed pixel stream,
  including its checksum and exact expected scanline size. Adam7 interlacing is
  supported. Expansion is checked with a bounded discarded output buffer.
- Pillow's image-size protection remains enabled. Malformed images, metadata
  warnings, and unsupported declarations fail with a fixed safe message.

## Pixel rules

1. Apply EXIF orientation once, including mirror orientations. Width and height
   are swapped for orientations 5–8. Missing orientation means 1.
2. Prefer a valid RGB ICC profile; convert it to sRGB using relative colorimetric
   intent and flags 0. PNG profiles are limited to 1 MiB after decompression.
3. Without a profile, accept an explicit PNG sRGB or EXIF sRGB declaration.
4. For truly untagged RGB, record the `assumed_srgb` policy.
5. Reject unsupported PNG cICP declarations, malformed color declarations, and
   gamma/chromaticity-only sources. Convert these to sRGB before retrying.
6. Transform alpha with orientation only. Color conversion never changes alpha.
7. Save RGB as RGB and RGBA as RGBA. Remove all source metadata, including EXIF,
   ICC profiles, comments, and file-related fields. The prepared PNG is untagged
   sRGB by this contract.

A temporary file is written in the output directory, flushed, reopened, and
compared with every prepared color and alpha channel. A no-overwrite hard link
publishes the verified file atomically. Failure keeps the source and any
existing output unchanged. Temporary files are removed on normal failure paths.
The output directory must support same-directory hard links.

## Python services and records

```python
from pathlib import Path
from backend_service.image_preparation import (
    inspect_image,
    prepare_image,
    read_prepared_png,
)

summary = inspect_image(Path("source.jpg"))
summary = prepare_image(Path("source.jpg"), Path("prepared.png"))
result = read_prepared_png(Path("prepared.png"))
image = result.image  # Owned Pillow RGB/RGBA image; integer channel values.
```

- Each source can be a `Path` or a binary stream. Streams are read from their
  current position and remain open for the caller.
- `ImageSummary` is a strict validated record: format, mode, original/prepared
  dimensions, input byte count, orientation, color policy, and pixel policy.
- `pixel_policy="prepare_srgb"` identifies source preparation.
- `read_prepared_png` returns `PreparedPixels(image, summary)` and accepts PNG
  only. It checks the input but performs **no orientation or color conversion**.
  Its `pixel_policy="preserve_stored"` makes this distinction explicit. Other
  summary fields describe source declarations; `color_policy` does not mean
  conversion happened in this read mode.
- The reader keeps exactly the stored integers and returns no source metadata.
- Failures use `ApplicationFailure` with `image_invalid`, `image_limits`,
  `image_color`, `image_write`, or `image_resources`. Memory shortages return
  `image_resources` with advice to free memory or use a smaller image.
  Error text never includes source values or
  operating-system details. Pillow metadata diagnostics are suppressed only in
  the current image-service context, even when application debug logging is
  enabled. Logging outside that context remains available.

## Verification

```sh
uv run python -m pytest tests/backend/test_image_preparation.py \
  tests/backend/test_image_validation.py tests/backend/test_image_color.py
```

Tests cover all eight orientations, RGB/RGBA pixels and alpha, real sixteen-bit
sources, ICC/sRGB precedence, malformed and unsupported declarations, complete
JPEG profile segments, both common 4K sizes, bounded unknown-size streams,
metadata removal, existing files/symlinks, publication races, and failed or
pixel-corrupt writes.

A local CPU measurement on 2026-09-23 used Python 3.12 and Pillow 12.3.0 on
macOS arm64. Preparing and reopening a 4096 × 2160 RGBA solid-color PNG took
0.1262 seconds wall time, 0.1221 seconds CPU time, with a process peak resident
memory of 204.55 MiB. The 39,909-byte input is compressible; this is a repeatable
smoke measurement, not a limit or a guarantee for all images. Peak memory also
includes interpreter imports and source-fixture generation.
