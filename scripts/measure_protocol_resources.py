"""Measure public synthetic protocol/image cases in separate CPU processes."""

import argparse
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Allow execution from a checkout without depending on the current directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASES = {"512": (512, 512), "1024": (1024, 1024), "4k": (4096, 2160)}


def measure(case: str) -> dict[str, str | int | float]:
    """Time exact recovery and image preparation with clearly public inputs."""
    import numpy as np
    from PIL import Image

    from backend_service.image_preparation import prepare_image, read_prepared_png
    from backend_service.message_protocol import decode_message, encode_message
    from backend_service.payload_capacity import calculate_capacity
    from backend_service.payload_map import create_payload_map, recover_payload_bytes
    from schemas.protocol import ProtocolContext

    width, height = CASES[case]
    context = ProtocolContext(width=width, height=height)
    maximum = calculate_capacity(context).maximum_message_bytes
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    message = "x" * maximum
    protected = encode_message(message, "PUBLIC BENCHMARK ONLY", context)
    payload = create_payload_map(protected, context)
    recovered = recover_payload_bytes(payload, context)
    if decode_message(recovered, "PUBLIC BENCHMARK ONLY", context) != message:
        raise RuntimeError("Public benchmark recovery failed.")
    protocol_wall = time.perf_counter() - wall_start
    protocol_cpu = time.process_time() - cpu_start
    with tempfile.TemporaryDirectory(prefix="stegolab_public_benchmark_") as directory:
        source, output = (
            Path(directory) / "source.png",
            Path(directory) / "prepared.png",
        )
        pixels = np.empty((height, width, 4), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(width, dtype=np.uint16) % 256
        pixels[:, :, 1] = (np.arange(height, dtype=np.uint16) % 256)[:, None]
        pixels[:, :, 2] = 64
        pixels[:, :, 3] = 127
        Image.fromarray(pixels).save(source)
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        prepare_image(source, output)
        prepared = read_prepared_png(output)
        if prepared.image.tobytes() != pixels.tobytes():
            raise RuntimeError("Public benchmark pixels changed.")
        image_wall = time.perf_counter() - wall_start
        image_cpu = time.process_time() - cpu_start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB. Include imports and fixture buffers.
    peak_mebibytes = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return {
        "case": case,
        "width": width,
        "height": height,
        "message_bytes": maximum,
        "protocol_wall_seconds": round(protocol_wall, 4),
        "protocol_cpu_seconds": round(protocol_cpu, 4),
        "image_wall_seconds": round(image_wall, 4),
        "image_cpu_seconds": round(image_cpu, 4),
        "peak_process_mebibytes": round(peak_mebibytes, 2),
    }


def main() -> None:
    """Print reproducible measurements without storing messages or images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    arguments = parser.parse_args()
    if arguments.case:
        print(json.dumps(measure(arguments.case)))
        return
    records = []
    for case in CASES:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", case],
            check=True,
            capture_output=True,
            text=True,
        )
        records.append(json.loads(result.stdout))
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "measurements": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
