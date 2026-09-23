"""Prepare one source image at a time and preserve its upstream provenance."""

import hashlib
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from backend_service.dataset_files import read_source_bytes
from backend_service.dataset_image import DatasetPreparedImage, prepare_dataset_image
from backend_service.dataset_inventory import DatasetInventory
from backend_service.dataset_storage import DatasetWriter
from backend_service.failures import ApplicationFailure
from schemas.datasets import DatasetImageRecord, DatasetRejection


def import_images(
    source: Path,
    inventory: DatasetInventory,
    writer: DatasetWriter,
    progress: Callable[[str], None],
) -> tuple[list[DatasetImageRecord], list[DatasetRejection]]:
    """Hash and decode the same bytes, without retaining source pixel buffers."""
    records: list[DatasetImageRecord] = []
    rejections = list(inventory.rejections)
    rejected = {record.source_path for record in rejections}
    cached: dict[str, DatasetPreparedImage] = {}
    (writer.stage / "images").mkdir()
    for index, item in enumerate(inventory.selected):
        if item.relative_path in rejected:
            continue
        data = read_source_bytes(source, item)
        checksum = hashlib.sha256(data).hexdigest()
        relative_output = f"images/{checksum}.png"
        try:
            if checksum not in cached:
                cached[checksum] = prepare_dataset_image(
                    data, writer.stage / relative_output, writer.reserve_write
                )
            image = cached[checksum]
        except ApplicationFailure as failure:
            if failure.code not in ("image_invalid", "image_color", "image_limits"):
                raise
            rejections.append(
                DatasetRejection(
                    source_path=item.relative_path,
                    source_bytes=item.size_bytes,
                    reason=failure.code,
                    source_checksum=checksum,
                    upstream_split=(
                        inventory.metadata.rows[
                            Path(item.relative_path).name
                        ].upstream_split
                        if inventory.metadata is not None
                        else None
                    ),
                )
            )
            continue
        finally:
            del data
        identity = None
        upstream_split = None
        subset = None
        if inventory.metadata is not None:
            row = inventory.metadata.rows[Path(item.relative_path).name]
            identity, upstream_split, subset = (
                row.upstream_identity,
                row.upstream_split,
                row.subset,
            )
        records.append(
            DatasetImageRecord.model_validate(
                {
                    **asdict(image),
                    "source_path": item.relative_path,
                    "prepared_path": relative_output,
                    "source_checksum": checksum,
                    "source_bytes": item.size_bytes,
                    "source_identity": identity,
                    "upstream_split": upstream_split,
                    "subset": subset,
                }
            )
        )
        if (index + 1) % 100 == 0:
            progress(f"dataset_images_prepared_{index + 1}")
    return records, rejections
