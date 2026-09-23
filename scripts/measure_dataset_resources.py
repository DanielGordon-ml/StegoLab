"""Measure the 32-million-pixel dataset boundary outside routine CI."""

import hashlib
import io
import json
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    """Probe a large real JPEG-to-PNG path using disposable seeded pixels."""
    from backend_service.dataset_image import (
        prepare_dataset_image,
        validate_dataset_png,
    )

    started = time.perf_counter()
    pixels = np.random.default_rng(0).integers(0, 256, (4000, 8000, 3), dtype=np.uint8)
    output = io.BytesIO()
    with Image.fromarray(pixels) as image:
        image.save(output, format="JPEG", quality=95)
    del pixels
    source = output.getvalue()
    output.close()
    if len(source) > 50 * 1024**2:
        raise RuntimeError("The public resource fixture exceeds its source limit.")
    reserved = 0

    def reserve(size: int) -> None:
        """Count encoded writes and enforce the per-file output ceiling."""
        nonlocal reserved
        reserved += size
        if reserved > 128 * 1024**2:
            raise RuntimeError("The public resource probe exceeded its output limit.")

    cache = Path(".cache")
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataset_probe_", dir=cache) as directory:
        destination = Path(directory) / "prepared.png"
        preparation_started = time.perf_counter()
        prepared = prepare_dataset_image(source, destination, reserve)
        preparation_seconds = time.perf_counter() - preparation_started
        verified = validate_dataset_png(destination)
        if verified != prepared:
            # Prepared provenance describes the JPEG; recovery describes its PNG.
            if (
                verified.rgb_checksum != prepared.rgb_checksum
                or verified.prepared_checksum != prepared.prepared_checksum
                or verified.prepared_bytes != prepared.prepared_bytes
            ):
                raise RuntimeError("The large prepared PNG did not verify.")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "fixture": "seed_0_rgb_noise_8000_4000_jpeg_quality_95",
                "platform": platform.platform(),
                "python": platform.python_version(),
                "source_bytes": len(source),
                "source_checksum": hashlib.sha256(source).hexdigest(),
                "prepared_bytes": prepared.prepared_bytes,
                "prepared_checksum": prepared.prepared_checksum,
                "rgb_checksum": prepared.rgb_checksum,
                "reserved_bytes": reserved,
                "preparation_seconds": round(preparation_seconds, 4),
                "total_seconds": round(time.perf_counter() - started, 4),
                "peak_process_mebibytes": round(
                    peak / (1024**2 if sys.platform == "darwin" else 1024), 2
                ),
                "verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
