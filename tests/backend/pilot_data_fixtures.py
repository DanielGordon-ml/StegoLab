"""Small real immutable revisions for lazy pilot loading and crop checks."""

import hashlib
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from backend_service.dataset_image import prepare_dataset_image
from backend_service.dataset_manifest import write_manifest
from schemas.datasets import DatasetImageRecord, DatasetPreparationRequest


def pilot_record(
    directory: Path,
    number: int,
    split: Literal["training", "validation", "test"],
    size: int,
) -> DatasetImageRecord:
    """Build inexpensive nonconstant pixels so crop restoration is observable."""
    (directory / "images").mkdir(exist_ok=True, parents=True)
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(size, dtype=np.uint16)[None, :] % 256
    pixels[:, :, 1] = np.arange(size, dtype=np.uint16)[:, None] % 256
    pixels[:, :, 2] = number
    stream = BytesIO()
    with Image.fromarray(pixels) as image:
        image.save(stream, format="PNG")
    content = stream.getvalue()
    relative = f"images/{number}.png"
    prepared = prepare_dataset_image(content, directory / relative, lambda size: None)
    return DatasetImageRecord(
        source_path=f"{split}/{number:03}.png",
        prepared_path=relative,
        source_identity=f"fixture:{number}",
        upstream_split=split,
        source_checksum=hashlib.sha256(content).hexdigest(),
        source_bytes=len(content),
        **asdict(prepared),
    )


def pilot_revision(directory: Path, tuning_count: int = 2) -> Path:
    """Write five training, native tuning, two excluded and one held-out cover."""
    records = [pilot_record(directory, number, "training", 280) for number in range(5)]
    records.extend(
        pilot_record(directory, number + 5, "validation", 1024)
        for number in range(tuning_count)
    )
    records.extend(
        pilot_record(directory, number + 5 + tuning_count, "validation", 300)
        for number in range(2)
    )
    records.append(pilot_record(directory, tuning_count + 7, "test", 280))
    manifest = write_manifest(
        directory,
        DatasetPreparationRequest(source_directory="unused"),
        records,
        [],
        before_write=lambda size: None,
        metadata_checksum="a" * 64,
        expected_images=len(records),
    )
    destination = directory.parent / manifest.revision
    directory.rename(destination)
    return destination
