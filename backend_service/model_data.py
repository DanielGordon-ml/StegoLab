"""Frozen small-sample model inputs with one full dataset validation per run."""

import hashlib
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from backend_service.dataset_reader import DatasetExample
from backend_service.dataset_serialization import contained_file
from backend_service.dataset_validation import load_manifest, load_validated_dataset
from backend_service.failures import ApplicationFailure
from schemas.datasets import DatasetManifest


@dataclass(frozen=True)
class ProofData:
    """Keep only small training tensors and frozen provenance in memory."""

    training_crops: tuple[Tensor, ...]
    tuning_crops: tuple[Tensor, ...]
    manifest: DatasetManifest
    identities: dict[str, tuple[str, ...]]
    tuning_examples: tuple[DatasetExample, ...]


def _data_failure() -> ApplicationFailure:
    """Give safe guidance without printing dataset paths or parser details."""
    return ApplicationFailure(
        "model_dataset_invalid",
        "The model sample is unavailable or changed. Validate the dataset and "
        "provide four training and four tuning images; tuning sides must be "
        "at least 1,024 pixels.",
        422,
    )


def load_center_crop(example: DatasetExample, width: int, height: int) -> Tensor:
    """Read verified prepared RGB values and make one explicit center crop."""
    try:
        if width < 1 or height < 1:
            raise ValueError
        with example.path.open("rb") as stream:
            content = stream.read(example.record.prepared_bytes + 1)
        if (
            len(content) != example.record.prepared_bytes
            or hashlib.sha256(content).hexdigest() != example.record.prepared_checksum
        ):
            raise ValueError
        with Image.open(BytesIO(content), formats=["PNG"]) as image:
            if (
                image.mode not in ("RGB", "RGBA")
                or image.size != (example.record.width, example.record.height)
                or width > image.width
                or height > image.height
            ):
                raise ValueError
            left, top = (image.width - width) // 2, (image.height - height) // 2
            with image.crop((left, top, left + width, top + height)) as cropped:
                with cropped.convert("RGB") as rgb:
                    pixels = np.array(rgb, dtype=np.uint8, copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float() / 255.0
    except (OSError, ValueError, MemoryError):
        raise _data_failure() from None


def load_proof_data(directory: Path, *, deadline: float | None = None) -> ProofData:
    """Validate once and select four representatives from each development split."""

    def check_progress() -> None:
        """Stop before another bounded file operation once its allowance is used."""
        if deadline is not None and time.monotonic() >= deadline:
            raise ApplicationFailure(
                "proof_deadline",
                "The CPU proof time limit was reached during dataset validation. "
                "Keep the last checkpoint and existing dataset files.",
            )

    check_progress()
    try:
        preview = load_manifest(directory)
    except (OSError, ValueError, MemoryError, ApplicationFailure):
        raise _data_failure() from None
    if preview.selected_images > 30 or preview.prepared_bytes > 512 * 1024**2:
        raise ApplicationFailure(
            "model_sample_limits",
            "Use a prepared proof sample with at most 30 source records and "
            "512 MiB of prepared images. The full corpus is outside this CPU proof.",
            422,
        )
    manifest, records = load_validated_dataset(directory, check_progress=check_progress)
    selected: dict[str, tuple[DatasetExample, ...]] = {}
    for split in ("train", "tuning"):
        examples = tuple(
            DatasetExample(record, contained_file(directory, record.prepared_path))
            for record in records
            if record.assigned_split == split
            and record.eligible
            and record.source_path == record.representative_source_path
        )[:4]
        if len(examples) != 4:
            raise _data_failure()
        selected[split] = examples
    if any(
        min(example.record.width, example.record.height) < 1024
        for example in selected["tuning"]
    ):
        raise _data_failure()
    crops: dict[str, tuple[Tensor, ...]] = {}
    for split, examples in selected.items():
        loaded: list[Tensor] = []
        for example in examples:
            check_progress()
            loaded.append(load_center_crop(example, 256, 256))
        crops[split] = tuple(loaded)
    check_progress()
    return ProofData(
        training_crops=crops["train"],
        tuning_crops=crops["tuning"],
        manifest=manifest,
        identities={
            split: tuple(
                example.record.source_identity or example.record.source_path
                for example in examples
            )
            for split, examples in selected.items()
        },
        tuning_examples=selected["tuning"],
    )
