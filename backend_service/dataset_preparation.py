"""Prepare and atomically publish reproducible local image datasets."""

import json
import os
import platform
import resource
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import __version__ as pillow_version
from PIL import features

from backend_service.dataset_import import import_images
from backend_service.dataset_inventory import DatasetInventory, inventory_dataset
from backend_service.dataset_manifest import write_manifest
from backend_service.dataset_reuse import find_reusable_dataset
from backend_service.dataset_storage import DatasetWriter
from backend_service.dataset_validation import validate_dataset
from schemas.datasets import DatasetPreparationRequest, DatasetSummary


def _estimate_prepared_bytes(inventory: DatasetInventory) -> int:
    """Estimate PNG expansion before import; every actual write remains bounded."""
    estimate = 0
    for item in inventory.selected:
        header = inventory.headers.get(item.relative_path)
        if header is None:
            continue
        channels = 4 if header.mode == "RGBA" else 3
        raw_bound = header.width * header.height * channels + header.height + 262144
        estimate += min(raw_bound, item.size_bytes * 8 + 65536)
    # Allow bounded records and manifest overhead independently of image compression.
    return estimate + len(inventory.selected) * 4096 + 1024**2


def _provenance() -> dict[str, str]:
    """Identify relevant supported encoders without machine-specific paths."""
    return {
        "application": "stegolab_0.1.0",
        "preparation": "dataset_v1",
        "python": platform.python_version(),
        "pillow": pillow_version,
        "jpeg": str(features.version("jpg")),
        "zlib": str(features.version("zlib")),
        "littlecms": str(features.version("littlecms2")),
    }


def _record_run(
    summary: DatasetSummary, inventory: DatasetInventory, started: float, estimate: int
) -> None:
    """Save measured run metadata separately from the immutable revision."""
    identifier = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_") + uuid4().hex
    directory = Path(os.environ.get("STEGOLAB_LOG_DIRECTORY", "logs")) / identifier
    directory.mkdir(parents=True, exist_ok=False)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "schema_version": 1,
        "event": "dataset_preparation_completed",
        "summary": summary.model_dump(mode="json"),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_process_mebibytes": round(
            peak / (1024**2 if sys.platform == "darwin" else 1024), 2
        ),
        "scanned_entries": inventory.source.scanned_entries,
        "scanned_bytes": inventory.source.source_bytes,
        "estimated_prepared_bytes": estimate,
        "source_modes": {
            mode: sum(header.mode == mode for header in inventory.headers.values())
            for mode in ("L", "RGB", "RGBA")
        },
        "limits": {
            "scanned_entries": 200_000,
            "source_bytes": 100 * 1024**3,
            "prepared_bytes": 100 * 1024**3,
            "source_file_bytes": 50 * 1024**2,
            "prepared_file_bytes": 128 * 1024**2,
            "maximum_pixels": 32_000_000,
            "maximum_side": 8192,
            "minimum_free_bytes": 10 * 1024**3,
            "minimum_free_fraction": 0.1,
        },
    }
    with (directory / "dataset_run.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare_dataset(
    request: DatasetPreparationRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> DatasetSummary:
    """Freeze local images and source splits, then validate before publication."""
    request = DatasetPreparationRequest.model_validate(request)
    report_progress = progress or (lambda event: None)
    started = time.perf_counter()
    with DatasetWriter(
        Path(request.output_root), Path(request.source_directory)
    ) as writer:
        report_progress("dataset_inventory_started")
        inventory = inventory_dataset(request)
        existing_summary = find_reusable_dataset(request, inventory, _provenance())
        if existing_summary is not None:
            _record_run(existing_summary, inventory, started, 0)
            report_progress("dataset_revision_reused")
            return existing_summary
        estimate = _estimate_prepared_bytes(inventory)
        writer.check_estimated_space(estimate)
        report_progress("dataset_preparation_started")
        records, rejections = import_images(
            Path(request.source_directory), inventory, writer, report_progress
        )
        metadata = inventory.metadata
        selected_by_split: dict[str, int] = {}
        if metadata is not None:
            for item in inventory.selected:
                row = metadata.rows.get(Path(item.relative_path).name)
                if row is not None:
                    selected_by_split[row.upstream_split] = (
                        selected_by_split.get(row.upstream_split, 0) + 1
                    )
        manifest = write_manifest(
            writer.stage,
            request,
            records,
            rejections,
            before_write=writer.reserve_write,
            metadata_checksum=metadata.sha256 if metadata else None,
            expected_images=len(metadata.rows) if metadata else None,
            discovered_images=sum(
                item.kind != "sidecar" for item in inventory.source.files
            ),
            source_provenance=_provenance(),
            expected_by_split=inventory.expected_by_split,
            discovered_by_split=inventory.discovered_by_split,
            selected_by_split=selected_by_split if metadata is not None else None,
        )
        report_progress("dataset_integrity_check_started")
        summary = validate_dataset(writer.stage, require_revision_name=False)
        relative = f"{request.dataset_name}/{manifest.revision}"
        existing = writer.reuse_path(relative)
        if existing is not None:
            summary = validate_dataset(existing).model_copy(update={"reused": True})
        else:
            writer.publish(relative)
        _record_run(summary, inventory, started, estimate)
        report_progress("dataset_preparation_completed")
        return summary
