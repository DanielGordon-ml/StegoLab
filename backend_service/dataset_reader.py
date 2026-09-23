"""Minimal offline training handoff without split fallback or augmentation."""

from dataclasses import dataclass
from pathlib import Path

from backend_service.dataset_serialization import contained_file
from backend_service.dataset_validation import load_validated_dataset
from schemas.dataset_common import DATASET_SPLITS, DatasetSplit
from schemas.datasets import DatasetImageRecord


@dataclass(frozen=True)
class DatasetExample:
    """Provide one eligible unique cover and its frozen provenance."""

    record: DatasetImageRecord
    path: Path


@dataclass(frozen=True)
class DatasetReaderResult:
    """Keep an empty requested split explicit rather than borrowing examples."""

    examples: tuple[DatasetExample, ...]
    warnings: tuple[str, ...]


def read_dataset(directory: Path, split: DatasetSplit) -> DatasetReaderResult:
    """Validate offline copies and select eligible representatives in one split."""
    if split not in DATASET_SPLITS:
        raise ValueError("The requested dataset split is unsupported.")
    _, records = load_validated_dataset(directory)
    examples = tuple(
        DatasetExample(record, contained_file(directory, record.prepared_path))
        for record in records
        if record.assigned_split == split
        and record.eligible
        and record.source_path == record.representative_source_path
    )
    warnings = () if examples else (f"No eligible unique examples in {split}.",)
    return DatasetReaderResult(examples=examples, warnings=warnings)
