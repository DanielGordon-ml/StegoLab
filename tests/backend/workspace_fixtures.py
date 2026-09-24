"""Small real metadata fixtures for the workspace discovery boundary."""

import hashlib
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image

from backend_service.dataset_image import prepare_dataset_image
from backend_service.dataset_manifest import write_manifest
from schemas.datasets import DatasetImageRecord, DatasetPreparationRequest
from schemas.evaluation import EvaluationReport, PngEvaluationTrial


def prepared_revision(root: Path) -> Path:
    """Prepare nine distinct public images with four training and four tuning."""
    stage = root / "datasets" / "example" / "stage"
    (stage / "images").mkdir(parents=True)
    records: list[DatasetImageRecord] = []
    for number in range(1, 10):
        stream = BytesIO()
        with Image.new("RGB", (1024, 1024), (number, number * 2, number * 3)) as image:
            image.save(stream, format="PNG")
        content = stream.getvalue()
        relative = f"images/{number}.png"
        prepared = prepare_dataset_image(content, stage / relative, lambda size: None)
        split: Literal["training", "validation", "test"] = (
            "training" if number < 5 else "validation" if number < 9 else "test"
        )
        records.append(
            DatasetImageRecord(
                source_path=f"{split}/{number:03}.png",
                prepared_path=relative,
                source_identity=f"uhd_iqa:{number}.png",
                upstream_split=split,
                source_checksum=hashlib.sha256(content).hexdigest(),
                source_bytes=len(content),
                **asdict(prepared),
            )
        )
    manifest = write_manifest(
        stage,
        DatasetPreparationRequest(source_directory="unused"),
        records,
        [],
        before_write=lambda size: None,
        metadata_checksum="a" * 64,
        expected_images=9,
    )
    destination = stage.parent / manifest.revision
    stage.rename(destination)
    return destination


def evaluation_report(compatibility: str, revision: str) -> EvaluationReport:
    """Build an explicit measured fixture, never a substitute for model evidence."""
    return EvaluationReport(
        kind="proof",
        compatibility_identifier=compatibility,
        dataset_revision=revision,
        completed=True,
        expected_bit_cases=0,
        bit_cases_completed=0,
        expected_png_trials=1,
        png_trials=[
            PngEvaluationTrial(
                source_identity="public",
                width=1024,
                height=1024,
                fixture_kind="maximum",
                recovered=True,
                elapsed_seconds=1.0,
            )
        ],
        exact_recovery_count=1,
        exact_recovery_rate=1.0,
        median_peak_signal_to_noise_ratio=26.22,
        median_structural_similarity=0.865,
        elapsed_seconds=1.0,
        peak_process_memory_bytes=100,
        learning_gate_passed=False,
        stopped_reason="complete",
    )
