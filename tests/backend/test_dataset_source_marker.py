"""Provenance markers beside fetched sources: reading, reuse and preparation."""

import os
from pathlib import Path

import pytest

from backend_service.dataset_serialization import canonical_json, checksum
from backend_service.dataset_sources.source_marker import (
    MARKER_NAME,
    MAXIMUM_MARKER_BYTES,
    METADATA_NAME,
    marker_provenance,
    preparation_document,
    raw_folder_state,
    read_source_marker,
    write_source_marker,
)
from backend_service.dataset_storage import DatasetWriter
from backend_service.failures import ApplicationFailure
from schemas.dataset_records import DatasetPreparationRequest
from schemas.dataset_sources import SourceMarker

IDENTITY = "a" * 64
OTHER_IDENTITY = "b" * 64


def completed_marker(**overrides: object) -> SourceMarker:
    """Build a completed https archive marker that tests then vary."""
    values: dict[str, object] = {
        "source_kind": "https_archive",
        "source_name": "fake_source",
        "reference": "https://example.org/fake.zip",
        "source_url": "https://example.org/fake.zip",
        "terms_reference": "https://example.org/terms",
        "resolved_revision": "etag-1",
        "materialization_identity": IDENTITY,
        "content": "images",
        "assets": [{"path": "fake.zip", "size_bytes": 1234, "sha256": "c" * 64}],
        "member_count": 4,
        "rejected_member_count": 1,
        "split_mapping": {"validation": "held_out", "training": "train"},
        "declared_splits": True,
        "created_at": "2026-09-27T00:00:00+00:00",
        "completed": True,
    }
    values.update(overrides)
    return SourceMarker.model_validate(values)


def place_marker(folder: Path, marker: SourceMarker) -> None:
    """Write a marker directly into an existing folder without a writer."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MARKER_NAME).write_bytes(canonical_json(marker.model_dump(mode="json")))


def remote_request(source: Path, **overrides: object) -> DatasetPreparationRequest:
    """Build a remote preparation request pointing at one source folder."""
    values: dict[str, object] = {
        "source_directory": str(source),
        "source_kind": "https_archive",
        "source_url": "https://example.org/fake.zip",
        "terms_reference": "https://example.org/terms",
        "dataset_name": "fake",
    }
    values.update(overrides)
    return DatasetPreparationRequest.model_validate(values)


def test_marker_round_trip_through_a_writer_stage(tmp_path: Path) -> None:
    """A marker written into a stage is read back exactly after publication."""
    data_root, cache_root = tmp_path / "data", tmp_path / "cache"
    data_root.mkdir()
    cache_root.mkdir()
    marker = completed_marker()
    with DatasetWriter(data_root, cache_root) as writer:
        write_source_marker(writer, marker)
        writer.publish("fake_source")
    assert read_source_marker(data_root / "fake_source") == marker
    assert read_source_marker(data_root / "absent") is None
    assert raw_folder_state(data_root, "fake_source", IDENTITY) == "reusable"


@pytest.mark.parametrize(
    "damage", ["garbage", "symlink", "oversized", "directory", "incomplete_json"]
)
def test_unreadable_markers_fail_without_details(tmp_path: Path, damage: str) -> None:
    """Broken, linked or oversized markers are reported with one fixed message."""
    folder = tmp_path / "source"
    folder.mkdir()
    path = folder / MARKER_NAME
    if damage == "garbage":
        path.write_bytes(b"not json")
    elif damage == "symlink":
        place_marker(tmp_path / "elsewhere", completed_marker())
        path.symlink_to(tmp_path / "elsewhere" / MARKER_NAME)
    elif damage == "oversized":
        path.write_bytes(b" " * (MAXIMUM_MARKER_BYTES + 1))
    elif damage == "directory":
        path.mkdir()
    else:
        path.write_bytes(b'{"schema_version": 1}')
    with pytest.raises(ApplicationFailure) as failure:
        read_source_marker(folder)
    assert failure.value.code == "source_marker_invalid"
    assert str(tmp_path) not in failure.value.message
    assert raw_folder_state(tmp_path, "source", IDENTITY) == "conflict"


def test_raw_folder_state_distinguishes_free_reusable_and_taken(
    tmp_path: Path,
) -> None:
    """Only a completed marker with the same identity allows folder reuse."""
    assert raw_folder_state(tmp_path, "missing", IDENTITY) == "available"
    (tmp_path / "foreign").mkdir()
    (tmp_path / "foreign" / "one.png").write_bytes(b"x")
    assert raw_folder_state(tmp_path, "foreign", IDENTITY) == "conflict"
    place_marker(tmp_path / "same", completed_marker())
    assert raw_folder_state(tmp_path, "same", IDENTITY) == "reusable"
    assert raw_folder_state(tmp_path, "same", OTHER_IDENTITY) == "conflict"
    place_marker(tmp_path / "partial", completed_marker(completed=False))
    assert raw_folder_state(tmp_path, "partial", IDENTITY) == "conflict"
    os.symlink(tmp_path / "same", tmp_path / "linked")
    assert raw_folder_state(tmp_path, "linked", IDENTITY) == "conflict"
    (tmp_path / "file").write_bytes(b"x")
    assert raw_folder_state(tmp_path, "file", IDENTITY) == "conflict"
    assert raw_folder_state(tmp_path, "../same", IDENTITY) == "conflict"


def test_marker_provenance_is_empty_for_local_kinds(tmp_path: Path) -> None:
    """UHD-IQA and local sources carry no fetched-source provenance."""
    local = DatasetPreparationRequest(
        source_directory=str(tmp_path), source_kind="local", dataset_name="local"
    )
    assert marker_provenance(local) == {}
    assert local.metadata_file is None
    uhd = DatasetPreparationRequest(source_directory=str(tmp_path))
    assert marker_provenance(uhd) == {}


def test_remote_provenance_requires_a_completed_marker_of_the_same_kind(
    tmp_path: Path,
) -> None:
    """Remote preparation refuses folders whose marker is absent or foreign."""
    folder = tmp_path / "source"
    folder.mkdir()
    with pytest.raises(ApplicationFailure) as failure:
        marker_provenance(remote_request(folder))
    assert failure.value.code == "source_marker_missing"
    place_marker(folder, completed_marker(completed=False))
    with pytest.raises(ApplicationFailure) as failure:
        marker_provenance(remote_request(folder))
    assert failure.value.code == "source_marker_missing"
    with pytest.raises(ApplicationFailure) as failure:
        marker_provenance(
            remote_request(folder, source_kind="upload", terms_reference="reviewed")
        )
    assert failure.value.code == "source_marker_missing"


def test_remote_provenance_freezes_marker_identity_and_checksums(
    tmp_path: Path,
) -> None:
    """The manifest provenance names the fetched source and hashes its lists."""
    folder = tmp_path / "source"
    marker = completed_marker()
    place_marker(folder, marker)
    provenance = marker_provenance(remote_request(folder))
    assets = [asset.model_dump(mode="json") for asset in marker.assets]
    assert provenance == {
        "source_name": "fake_source",
        "source_reference": "https://example.org/fake.zip",
        "source_revision": "etag-1",
        "source_materialization": IDENTITY,
        "source_assets_checksum": checksum(canonical_json(assets)),
        "source_members_checksum": checksum(
            canonical_json({"member_count": 4, "rejected_member_count": 1})
        ),
        "training_intended": "true",
    }
    evaluation = marker_provenance(remote_request(folder, training_intended=False))
    assert evaluation["training_intended"] == "false"
    assert all(isinstance(value, str) for value in provenance.values())


def test_preparation_document_follows_the_marker_or_stays_local(
    tmp_path: Path,
) -> None:
    """A fetched folder prepares as its source kind; other folders stay local."""
    output_root = tmp_path / "datasets"
    plain = tmp_path / "plain"
    plain.mkdir()
    document = preparation_document(plain, "plain_set", output_root)
    assert document == {
        "schema_version": 1,
        "source_kind": "local",
        "metadata_file": None,
        "source_directory": str(plain),
        "output_root": str(output_root),
        "dataset_name": "plain_set",
        "seed": 0,
    }
    request = DatasetPreparationRequest.model_validate(document)
    assert (request.source_url, request.terms_reference) == ("local", "not_reviewed")
    fetched = tmp_path / "fetched"
    place_marker(fetched, completed_marker())
    document = preparation_document(fetched, "fetched_set", output_root)
    assert document == {
        "schema_version": 1,
        "source_kind": "https_archive",
        "metadata_file": METADATA_NAME,
        "source_directory": str(fetched),
        "output_root": str(output_root),
        "dataset_name": "fetched_set",
        "seed": 0,
        "source_url": "https://example.org/fake.zip",
        "terms_reference": "https://example.org/terms",
        "split_mapping": {"validation": "held_out", "training": "train"},
        "training_intended": True,
    }
    request = DatasetPreparationRequest.model_validate(document)
    assert request.split_mapping == {"validation": "held_out", "training": "train"}
    unlabelled = tmp_path / "unlabelled"
    place_marker(unlabelled, completed_marker(declared_splits=False, split_mapping={}))
    document = preparation_document(unlabelled, "unlabelled_set", output_root)
    assert document["metadata_file"] is None and document["split_mapping"] == {}


@pytest.mark.parametrize(
    "changes",
    [
        {"source_kind": "uhd_iqa", "split_mapping": {"training": "train"}},
        {"source_kind": "local", "split_mapping": {"training": "train"}},
        {"source_kind": "local", "metadata_file": "labels.csv"},
        {"split_mapping": {"Bad Label": "train"}},
        {"split_mapping": {"training": "train", "9nine": "tuning"}},
    ],
)
def test_split_mappings_are_checked_against_the_source_kind(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    """Fixed UHD-IQA mapping, generated local splits and safe labels are enforced."""
    with pytest.raises(ValueError):
        remote_request(tmp_path, **changes)
    assert remote_request(tmp_path, split_mapping={}).split_mapping == {}
