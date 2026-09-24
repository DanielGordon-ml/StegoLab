"""Check the frozen reader handoff, split isolation, and crop integrity."""

import hashlib
import time
from collections.abc import Callable
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
import torch
from PIL import Image

from backend_service import dataset_validation, model_data
from backend_service.dataset_image import prepare_dataset_image
from backend_service.dataset_manifest import write_manifest
from backend_service.dataset_reader import DatasetExample
from backend_service.dataset_serialization import file_checksum
from backend_service.dataset_validation import load_validated_dataset
from backend_service.failures import ApplicationFailure
from backend_service.model_data import load_center_crop, load_proof_data
from schemas.datasets import (
    DatasetImageRecord,
    DatasetManifest,
    DatasetPreparationRequest,
)


def _record(
    directory: Path,
    number: int,
    split: Literal["training", "validation", "test"],
) -> DatasetImageRecord:
    """Create a unique, inexpensive source with explicit official split metadata."""
    directory.joinpath("images").mkdir(parents=True, exist_ok=True)
    stream = BytesIO()
    with Image.new("RGB", (1024, 1024), (number, number * 2, number * 3)) as image:
        image.save(stream, format="PNG")
    content = stream.getvalue()
    relative = f"images/{number}.png"
    prepared = prepare_dataset_image(content, directory / relative, lambda size: None)
    return DatasetImageRecord(
        source_path=f"{split}/{number:03}.png",
        prepared_path=relative,
        source_identity=f"uhd_iqa:{number}.png",
        upstream_split=split,
        source_checksum=hashlib.sha256(content).hexdigest(),
        source_bytes=len(content),
        **asdict(prepared),
    )


def test_validate_once_select_frozen_splits_and_detect_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One integrity pass serves both splits, and later crop reads reject tampering."""
    stage = tmp_path / "stage"
    records = [
        _record(stage, number, "training" if number < 5 else "validation")
        for number in range(1, 9)
    ]
    records.append(_record(stage, 9, "test"))
    manifest = write_manifest(
        stage,
        DatasetPreparationRequest(source_directory="unused"),
        records,
        [],
        before_write=lambda size: None,
        metadata_checksum="a" * 64,
        expected_images=9,
    )
    directory = tmp_path / manifest.revision
    stage.rename(directory)
    calls = 0

    def counted_validation(
        path: Path,
        *,
        check_progress: Callable[[], None] | None = None,
    ) -> tuple[DatasetManifest, list[DatasetImageRecord]]:
        """Count full validation calls without changing their real behavior."""
        nonlocal calls
        calls += 1
        return load_validated_dataset(path, check_progress=check_progress)

    monkeypatch.setattr(model_data, "load_validated_dataset", counted_validation)
    data = load_proof_data(directory)
    assert calls == 1
    assert data.identities == {
        "train": tuple(f"uhd_iqa:{number}.png" for number in range(1, 5)),
        "tuning": tuple(f"uhd_iqa:{number}.png" for number in range(5, 9)),
    }
    assert all(crop.shape == (3, 256, 256) for crop in data.training_crops)
    assert data.training_crops[0].dtype == torch.float32
    example = data.tuning_examples[0]
    assert load_center_crop(example, 513, 517).shape == (3, 517, 513)
    example.path.write_bytes(b"changed prepared output")
    with pytest.raises(ApplicationFailure, match="sample is unavailable or changed"):
        load_center_crop(example, 512, 512)


def test_center_crop_preserves_rgb_values_and_rejects_enlargement(
    tmp_path: Path,
) -> None:
    """Place a known center pixel in the exact crop position without interpolation."""
    record = _record(tmp_path, 1, "training")
    path = tmp_path / record.prepared_path
    with Image.open(path) as opened:
        image = opened.copy()
    with image:
        image.putpixel((512, 512), (255, 0, 128))
        image.save(path)
    content = path.read_bytes()
    updated = record.model_copy(
        update={
            "prepared_checksum": hashlib.sha256(content).hexdigest(),
            "prepared_bytes": len(content),
        }
    )
    example = DatasetExample(updated, path)
    crop = load_center_crop(example, 256, 256)
    assert torch.equal(crop[:, 128, 128], torch.tensor([1.0, 0.0, 128 / 255]))
    with pytest.raises(ApplicationFailure):
        load_center_crop(example, 1025, 512)


def test_expired_deadline_prevents_even_manifest_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject an exhausted caller before starting dataset work."""

    def unexpected_manifest(path: Path) -> DatasetManifest:
        """Fail the test if dataset reading begins after the deadline."""
        raise AssertionError("An expired operation must not read the dataset.")

    monkeypatch.setattr(model_data, "load_manifest", unexpected_manifest)
    with pytest.raises(ApplicationFailure) as caught:
        load_proof_data(tmp_path, deadline=time.monotonic() - 1.0)
    assert caught.value.code == "proof_deadline"


@pytest.mark.parametrize(
    "change", [{"selected_images": 31}, {"prepared_bytes": 512 * 1024**2 + 1}]
)
def test_oversized_sample_fails_before_any_integrity_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, int]
) -> None:
    """A manifest count or byte limit rejects full corpora before pixel decoding."""
    record = _record(tmp_path, 1, "training")
    manifest = write_manifest(
        tmp_path,
        DatasetPreparationRequest(source_directory="unused"),
        [record],
        [],
        before_write=lambda size: None,
        metadata_checksum="a" * 64,
        expected_images=1,
    )

    def preview(path: Path) -> DatasetManifest:
        """Supply an oversized manifest to isolate the inexpensive preflight gate."""
        return manifest.model_copy(update=change)

    def forbidden_validation(
        path: Path,
        *,
        check_progress: Callable[[], None] | None = None,
    ) -> tuple[DatasetManifest, list[DatasetImageRecord]]:
        """Detect a full scan that should have been prevented by the manifest."""
        raise AssertionError("An oversized proof sample must not be decoded.")

    monkeypatch.setattr(model_data, "load_manifest", preview)
    monkeypatch.setattr(model_data, "load_validated_dataset", forbidden_validation)
    with pytest.raises(ApplicationFailure) as caught:
        load_proof_data(tmp_path)
    assert caught.value.code == "model_sample_limits"


def test_progress_stop_prevents_next_prepared_file_and_keeps_failure_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check between prepared images without relabelling a timeout as corruption."""
    stage = tmp_path / "stage"
    records = [_record(stage, number, "training") for number in range(1, 4)]
    manifest = write_manifest(
        stage,
        DatasetPreparationRequest(source_directory="unused"),
        records,
        [],
        before_write=lambda size: None,
        metadata_checksum="a" * 64,
        expected_images=3,
    )
    directory = tmp_path / manifest.revision
    stage.rename(directory)
    progress_calls = 0
    checked_files = 0

    def check_progress() -> None:
        """Expire immediately before the second prepared image begins."""
        nonlocal progress_calls
        progress_calls += 1
        if progress_calls == 3:
            raise ApplicationFailure("proof_deadline", "The time limit was reached.")

    def count_checksum(path: Path, maximum_bytes: int) -> tuple[str, int]:
        """Count real prepared-file checks while preserving their implementation."""
        nonlocal checked_files
        checked_files += 1
        return file_checksum(path, maximum_bytes)

    monkeypatch.setattr(dataset_validation, "file_checksum", count_checksum)
    with pytest.raises(ApplicationFailure) as caught:
        load_validated_dataset(directory, check_progress=check_progress)
    assert caught.value.code == "proof_deadline"
    assert progress_calls == 3 and checked_files == 1
