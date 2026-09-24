"""Measure up to 16 shuffled training crops on CPU without running a model."""

import argparse
import json
import platform
import resource
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _signature(path: Path) -> tuple[int, int, int]:
    """Observe size and write-related timestamps without opening image pixels."""
    status = path.stat()
    return status.st_size, status.st_mtime_ns, status.st_ctime_ns


def main() -> int:
    """Write a timestamped log report after one bounded training-loader pass."""
    from backend_service.failures import ApplicationFailure
    from backend_service.pilot_data import load_pilot_data
    from backend_service.pilot_sampler import PilotSampler
    from schemas.pilot_loading import PilotLoadingReport

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--count", type=int, default=16, choices=range(1, 17))
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    arguments = parser.parse_args()
    if not 0 < arguments.deadline_seconds <= 120:
        parser.error("The deadline must be greater than zero and at most 120 seconds.")
    output_root = arguments.output_root.resolve()
    if not output_root.is_relative_to(Path("logs").resolve()):
        parser.error("Loading-probe reports must be saved under the logs directory.")
    dataset = arguments.dataset_directory.resolve()
    if output_root.is_relative_to(dataset):
        parser.error("Loading reports cannot be saved inside the dataset revision.")
    torch.set_num_threads(arguments.threads)
    started = time.monotonic()
    deadline = started + arguments.deadline_seconds
    sampler: PilotSampler | None = None
    revision = None
    selection = None
    signatures: dict[Path, tuple[int, int, int]] = {}
    shapes: list[tuple[int, int, int, int]] = []
    identities: list[str] = []
    error_code = None

    def expired(number: int, frame: FrameType | None) -> None:
        """Interrupt an overlong local probe without launching any extra work."""
        raise ApplicationFailure(
            "pilot_deadline", "The loading probe deadline expired."
        )

    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, arguments.deadline_seconds)
    try:
        data = load_pilot_data(dataset, deadline=deadline)
        revision = data.manifest.revision
        selection = data.selection.selection_checksum
        sampler = PilotSampler(data, 0, deadline=deadline)
        count = min(arguments.count, len(data.training_examples))
        files = [
            dataset / name
            for name in ("manifest.json", "records.jsonl", "rejections.jsonl")
        ]
        files.extend(
            data.training_examples[index].path for index in sampler.order[:count]
        )
        signatures = {path: _signature(path) for path in files}
        while sampler.consumed < count:
            crops = sampler.next_batch(min(4, count - sampler.consumed))
            shapes.append(
                (crops.shape[0], crops.shape[1], crops.shape[2], crops.shape[3])
            )
            identities.extend(sample.source_identity for sample in sampler.last_batch)
            del crops
    except ApplicationFailure as failure:
        error_code = failure.code
    except (OSError, ValueError, RuntimeError, MemoryError):
        error_code = "pilot_loading_failed"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    try:
        unchanged = bool(signatures) and all(
            _signature(path) == before for path, before in signatures.items()
        )
    except OSError:
        unchanged = False
    if not unchanged and error_code is None:
        error_code = "pilot_dataset_changed"
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = PilotLoadingReport(
        status="passed" if error_code is None else "failed",
        error_code=error_code,
        dataset_revision=revision,
        selection_checksum=selection,
        threads=arguments.threads,
        requested_count=arguments.count,
        consumed_count=sampler.consumed if sampler is not None else 0,
        crop_shapes=shapes,
        source_identities=identities,
        deadline_seconds=arguments.deadline_seconds,
        elapsed_seconds=round(time.monotonic() - started, 4),
        peak_process_mebibytes=round(
            peak / (1024**2 if sys.platform == "darwin" else 1024), 2
        ),
        immutable_file_metadata_unchanged=unchanged,
        platform=platform.platform(),
        python=platform.python_version(),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(UTC).strftime("pilot_loading_%Y%m%dT%H%M%S_%fZ.json")
    destination = output_root / name
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(report.model_dump_json(indent=2) + "\n")
    print(json.dumps({"status": report.status, "report": str(destination)}))
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
