"""Exercise leakage controls and source-independent immutable dataset records."""

import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image
from pydantic import ValidationError

from backend_service.dataset_groups import assign_groups
from backend_service.dataset_image import prepare_dataset_image
from backend_service.dataset_manifest import manifest_revision, write_manifest
from backend_service.dataset_reader import read_dataset
from backend_service.dataset_serialization import canonical_json, checksum
from backend_service.dataset_validation import inspect_dataset, validate_dataset
from backend_service.failures import ApplicationFailure
from schemas.datasets import DatasetImageRecord, DatasetPreparationRequest


def record(
    directory: Path,
    name: str,
    color: tuple[int, int, int] = (12, 24, 36),
    *,
    split: Literal["training", "validation", "test"] = "training",
) -> DatasetImageRecord:
    """Prepare one small real PNG entirely from an in-memory source."""
    (directory / "images").mkdir(exist_ok=True, parents=True)
    stream = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(stream, format="PNG")
    data = stream.getvalue()
    destination = directory / "images" / f"{name}.png"
    prepared = prepare_dataset_image(data, destination, lambda count: None)
    return DatasetImageRecord(
        source_path=f"{name}.png",
        prepared_path=f"images/{name}.png",
        source_identity=f"uhd_iqa:{name}.png",
        upstream_split=split,
        subset="test_fixture",
        source_checksum=hashlib.sha256(data).hexdigest(),
        source_bytes=len(data),
        **asdict(prepared),
    )


def publish(directory: Path, records: list[DatasetImageRecord]) -> Path:
    """Publish a minimal canonical fixture under its computed revision name."""
    request = DatasetPreparationRequest(source_directory="unused")
    manifest = write_manifest(
        directory,
        request,
        records,
        [],
        before_write=lambda count: None,
        metadata_checksum="1" * 64,
        expected_images=len(records),
    )
    revision = directory.parent / manifest.revision
    directory.rename(revision)
    return revision


def rewrite_manifest(directory: Path, change: dict[str, object]) -> None:
    """Recompute revision hashing to test structural checks beyond checksums."""
    values = json.loads((directory / "manifest.json").read_bytes())
    values.update(change)
    values["revision"] = manifest_revision(values)
    (directory / "manifest.json").write_bytes(canonical_json(values))


def test_canonical_revisions_ignore_source_root_and_record_order(
    tmp_path: Path,
) -> None:
    """Equivalent prepared inputs produce exactly the same frozen metadata."""
    directories = [tmp_path / name for name in ("first", "second")]
    revisions = []
    for index, directory in enumerate(directories):
        records = [record(directory, "b", (1, 2, 3)), record(directory, "a")]
        request = DatasetPreparationRequest(source_directory=f"/unrelated/{index}")
        manifest = write_manifest(
            directory,
            request,
            records[:: 1 if index else -1],
            [],
            before_write=lambda count: None,
            metadata_checksum="1" * 64,
            expected_images=2,
        )
        revisions.append(manifest.revision)
    assert revisions[0] == revisions[1]
    assert (directories[0] / "records.jsonl").read_bytes() == (
        directories[1] / "records.jsonl"
    ).read_bytes()


def test_linked_identity_chains_group_before_split_assignment(tmp_path: Path) -> None:
    """RGB duplicates and shared identities create one transitive split group."""
    a = record(tmp_path, "a")
    b = record(tmp_path, "b").model_copy(update={"source_identity": "same"})
    c = record(tmp_path, "c", (2, 3, 4)).model_copy(update={"source_identity": "same"})
    grouped = assign_groups([c, b, a], "uhd_iqa")
    assert len({item.split_group for item in grouped}) == 1
    assert grouped[1].representative_source_path == "a.png"
    assert grouped[2].representative_source_path == "c.png"
    conflicting = c.model_copy(update={"upstream_split": "test"})
    with pytest.raises(ApplicationFailure, match="conflict"):
        assign_groups([a, b, conflicting], "uhd_iqa")


def test_offline_reader_filters_duplicate_aliases_and_empty_splits(
    tmp_path: Path,
) -> None:
    """A completed revision works without any source and never borrows splits."""
    stage = tmp_path / "stage"
    path = publish(stage, [record(stage, "b"), record(stage, "a")])
    summary = validate_dataset(path)
    assert summary.integrity == "verified"
    assert summary.accepted_count == 2
    assert summary.duplicate_count == 1
    assert inspect_dataset(path).integrity == "not_checked"
    result = read_dataset(path, "train")
    assert [item.record.source_path for item in result.examples] == ["a.png"]
    assert result.examples[0].path.is_file()
    empty = read_dataset(path, "held_out")
    assert not empty.examples and empty.warnings


@pytest.mark.parametrize("damage", ["missing", "changed", "symlink", "orphan"])
def test_prepared_inventory_and_bytes_detect_damage(
    tmp_path: Path, damage: str
) -> None:
    """Missing, modified, linked, and untracked prepared outputs cannot validate."""
    stage = tmp_path / "stage"
    path = publish(stage, [record(stage, "a")])
    image = path / "images/a.png"
    if damage == "missing":
        image.unlink()
    elif damage == "changed":
        image.write_bytes(image.read_bytes() + b"changed")
    elif damage == "symlink":
        external = tmp_path / "external.png"
        image.rename(external)
        image.symlink_to(external)
    else:
        (path / "images/orphan.png").write_bytes(image.read_bytes())
    with pytest.raises(ApplicationFailure, match="integrity"):
        validate_dataset(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("assigned_split", "held_out"),
        ("split_group", "f" * 64),
        ("representative_source_path", "missing.png"),
        ("eligible", False),
        ("prepared_path", "../outside.png"),
    ],
)
def test_rehashed_tampering_cannot_change_record_invariants(
    tmp_path: Path, field: str, value: object
) -> None:
    """Rehashing metadata does not bypass grouping, paths, or eligibility rules."""
    stage = tmp_path / "stage"
    path = publish(stage, [record(stage, "a")])
    values = json.loads((path / "records.jsonl").read_bytes())
    values[field] = value
    data = canonical_json(values)
    (path / "records.jsonl").write_bytes(data)
    rewrite_manifest(path, {"records_checksum": checksum(data)})
    with pytest.raises(ApplicationFailure, match="integrity"):
        validate_dataset(path, require_revision_name=False)


def test_shared_prepared_files_count_physical_storage_once(tmp_path: Path) -> None:
    """Identical source-byte aliases may share one verified immutable PNG."""
    stage = tmp_path / "stage"
    a = record(stage, "a")
    b = a.model_copy(update={"source_path": "b.png", "source_identity": "uhd_iqa:b"})
    path = publish(stage, [a, b])
    summary = validate_dataset(path)
    assert summary.prepared_bytes == a.prepared_bytes
    assert summary.source_bytes == 2 * a.source_bytes
    assert summary.duplicate_count == 1


def test_local_seeded_splits_are_repeatable_and_unlabelled(tmp_path: Path) -> None:
    """Generated splits depend on frozen group identity and seed, not traversal."""
    first = record(tmp_path, "a").model_copy(update={"upstream_split": None})
    records = [
        first.model_copy(
            update={
                "source_path": f"{index}.png",
                "source_identity": None,
                "rgb_checksum": hashlib.sha256(str(index).encode()).hexdigest(),
            }
        )
        for index in range(40)
    ]
    a = assign_groups(records, "local", seed=123)
    b = assign_groups(records[::-1], "local", seed=123)
    assert a == b
    assert {item.assigned_split for item in a} == {"train", "tuning", "held_out"}


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 1.0},
        {"source_directory": 12},
        {"unknown": True},
        {"seed": True},
        {"selection": ["../bad"], "selection_name": "bad"},
        {"selection": ["ok.png", "ok.png"], "selection_name": "duplicates"},
        {"selection": ["ok.png"]},
        {"metadata_file": None},
    ],
)
def test_requests_are_strict_and_named_selections_are_safe(
    changes: dict[str, object],
) -> None:
    """Unsafe paths, implicit coercion, and unnamed or duplicate subsets fail."""
    with pytest.raises(ValidationError):
        DatasetPreparationRequest.model_validate(
            {"source_directory": "source", **changes}
        )


@pytest.mark.parametrize(
    "change",
    [
        {"source_bytes": 999},
        {"source_format": "JPEG"},
        {"source_mode": "L", "grayscale_converted": True},
        {"orientation": 2},
        {"color_policy": "declared_srgb"},
    ],
)
def test_same_source_checksum_requires_consistent_provenance(
    tmp_path: Path,
    change: dict[str, object],
) -> None:
    """Equal source byte identities cannot claim contradictory handling decisions."""
    stage = tmp_path / "stage"
    path = publish(stage, [record(stage, "a"), record(stage, "b")])
    rows = [
        json.loads(line) for line in (path / "records.jsonl").read_bytes().splitlines()
    ]
    rows[1].update(change)
    data = b"".join(canonical_json(row) for row in rows)
    (path / "records.jsonl").write_bytes(data)
    rewrite_manifest(
        path,
        {
            "records_checksum": checksum(data),
            "source_bytes": sum(row["source_bytes"] for row in rows),
        },
    )
    with pytest.raises(ApplicationFailure, match="integrity"):
        validate_dataset(path, require_revision_name=False)


def test_same_source_checksum_requires_consistent_prepared_pixels(
    tmp_path: Path,
) -> None:
    """Different prepared RGB covers cannot claim the same original source bytes."""
    stage = tmp_path / "stage"
    path = publish(stage, [record(stage, "a"), record(stage, "b", (90, 23, 45))])
    rows = [
        json.loads(line) for line in (path / "records.jsonl").read_bytes().splitlines()
    ]
    rows[1]["source_checksum"] = rows[0]["source_checksum"]
    rows[1]["source_bytes"] = rows[0]["source_bytes"]
    data = b"".join(canonical_json(row) for row in rows)
    (path / "records.jsonl").write_bytes(data)
    rewrite_manifest(
        path,
        {
            "records_checksum": checksum(data),
            "source_bytes": sum(row["source_bytes"] for row in rows),
        },
    )
    with pytest.raises(ApplicationFailure, match="integrity"):
        validate_dataset(path, require_revision_name=False)


def test_local_sources_cannot_override_generated_splits(tmp_path: Path) -> None:
    """Explicit source maps are unavailable for the current local request format."""
    image = record(tmp_path, "a")
    with pytest.raises(ApplicationFailure, match="split or duplicate"):
        assign_groups([image], "local")
