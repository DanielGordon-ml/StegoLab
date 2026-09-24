"""Metadata-only pilot preparation and deterministic tuning cover selection."""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from backend_service.dataset_reader import DatasetExample
from backend_service.dataset_serialization import canonical_json, contained_file
from backend_service.dataset_validation import validated_records
from backend_service.failures import ApplicationFailure
from schemas.datasets import DatasetImageRecord, DatasetManifest
from schemas.pilot_data import PilotCover, PilotExcludedCover, PilotSelection

FULL_UHD_REVISION = "75bef6dab2b3de136d1c5c359d6753a25e4a750d5387fe75921e394482f0776c"
FULL_UHD_COUNTS = {"train": 4269, "tuning": 904, "held_out": 900}


@dataclass(frozen=True)
class PilotData:
    """Reference immutable files and metadata without retaining decoded images."""

    manifest: DatasetManifest
    training_examples: tuple[DatasetExample, ...]
    tuning_examples: tuple[DatasetExample, ...]
    selection: PilotSelection
    identities: dict[str, tuple[str, ...]]


def pilot_data_failure() -> ApplicationFailure:
    """Keep dataset paths and parser details out of user-facing failures."""
    return ApplicationFailure(
        "pilot_dataset_invalid",
        "The pilot dataset is unavailable or changed. Check its immutable revision "
        "and prepared files; training requires 256-pixel sides and tuning 1,024.",
        422,
    )


def check_pilot_deadline(deadline: float | None) -> None:
    """Stop between bounded metadata and image operations after the deadline."""
    if deadline is not None and time.monotonic() >= deadline:
        raise ApplicationFailure(
            "pilot_deadline",
            "The pilot deadline was reached. Keep its last checkpoint.",
        )


def source_identity(record: DatasetImageRecord) -> str:
    """Use upstream identity when present and the frozen source path otherwise."""
    return record.source_identity or record.source_path


def _cover(record: DatasetImageRecord) -> PilotCover:
    """Freeze evaluation provenance without reading the image content."""
    return PilotCover(
        source_identity=source_identity(record),
        prepared_checksum=record.prepared_checksum,
        width=record.width,
        height=record.height,
    )


def _checksum(value: object) -> str:
    """Hash canonical structured values rather than ambiguous joined strings."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _select(
    manifest: DatasetManifest, records: list[DatasetImageRecord]
) -> tuple[list[DatasetImageRecord], list[DatasetImageRecord], PilotSelection]:
    """Select all training representatives and up to 32 native tuning covers."""
    unique = [
        record
        for record in records
        if record.eligible and record.source_path == record.representative_source_path
    ]
    training = [record for record in unique if record.assigned_split == "train"]
    tuning = [record for record in unique if record.assigned_split == "tuning"]
    eligible = [record for record in tuning if min(record.width, record.height) >= 1024]
    excluded = [record for record in tuning if min(record.width, record.height) < 1024]
    selected = sorted(
        eligible,
        key=lambda record: (
            _checksum(["pilot_data_v1", manifest.revision, source_identity(record)]),
            record.source_path,
        ),
    )[:32]
    if not training or not selected:
        raise pilot_data_failure()
    reasons = []
    if manifest.revision != FULL_UHD_REVISION:
        reasons.append("The GPU profile requires the frozen full UHD-IQA revision.")
    if manifest.source_kind != "uhd_iqa" or not manifest.full_coverage:
        reasons.append("The GPU profile requires complete UHD-IQA source coverage.")
    if (
        manifest.split_counts != FULL_UHD_COUNTS
        or manifest.unique_eligible_by_split != FULL_UHD_COUNTS
    ):
        reasons.append("The GPU profile requires 4,269/904/900 unique split covers.")
    if len(eligible) != 902 or len(excluded) != 2:
        reasons.append(
            "The GPU profile requires 902 native tuning covers and 2 exclusions."
        )
    covers = [_cover(record) for record in selected]
    selection = PilotSelection(
        dataset_revision=manifest.revision,
        records_checksum=manifest.records_checksum,
        split_counts=manifest.split_counts,
        training_count=len(training),
        tuning_eligible_count=len(eligible),
        tuning_selected=covers,
        tuning_excluded=[
            PilotExcludedCover(**_cover(record).model_dump()) for record in excluded
        ],
        training_checksum=_checksum(
            [_cover(record).model_dump(mode="json") for record in training]
        ),
        tuning_checksum=_checksum([cover.model_dump(mode="json") for cover in covers]),
        selection_checksum="0" * 64,
        gpu_profile_eligible=not reasons,
        gpu_profile_reasons=reasons,
    )
    values = selection.model_dump(mode="json", exclude={"selection_checksum"})
    selection.selection_checksum = _checksum(values)
    return training, selected, selection


def load_pilot_data(directory: Path, *, deadline: float | None = None) -> PilotData:
    """Verify canonical records and inventory without decoding any split images."""
    check_pilot_deadline(deadline)
    try:
        manifest, records = validated_records(directory)
        check_pilot_deadline(deadline)
        training, tuning, selection = _select(manifest, records)
        examples = {
            split: tuple(
                DatasetExample(record, contained_file(directory, record.prepared_path))
                for record in selected
            )
            for split, selected in (("train", training), ("tuning", tuning))
        }
        identities: dict[str, tuple[str, ...]] = {
            split: tuple(source_identity(example.record) for example in selected)
            for split, selected in examples.items()
        }
    except ApplicationFailure as failure:
        if failure.code == "pilot_deadline":
            raise
        raise pilot_data_failure() from None
    except (OSError, ValueError, MemoryError):
        raise pilot_data_failure() from None
    check_pilot_deadline(deadline)
    return PilotData(
        manifest=manifest,
        training_examples=examples["train"],
        tuning_examples=examples["tuning"],
        selection=selection,
        identities=identities,
    )
