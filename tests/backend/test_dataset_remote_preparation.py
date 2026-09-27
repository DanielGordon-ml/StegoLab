"""Prepare a materialized remote source folder end to end without a network."""

from pathlib import Path

import pytest
from PIL import Image
from test_dataset_source_marker import completed_marker, place_marker, remote_request

from backend_service.dataset_preparation import prepare_dataset
from backend_service.dataset_serialization import read_records
from backend_service.dataset_validation import load_manifest, validate_dataset
from backend_service.failures import ApplicationFailure
from schemas.dataset_common import SPLIT_MAPPING as DEFAULT_SPLIT_MAPPING
from schemas.dataset_common import DatasetSplit
from schemas.dataset_records import DatasetPreparationRequest
from schemas.datasets import DatasetImageRecord

SPLIT_MAPPING: dict[str, DatasetSplit] = {"validation": "held_out", "training": "train"}
PROVENANCE_KEYS = {
    "source_name",
    "source_reference",
    "source_revision",
    "source_materialization",
    "source_assets_checksum",
    "source_members_checksum",
    "training_intended",
}


def materialized_folder(
    root: Path, name: str, labels: dict[str, int], *, with_metadata: bool
) -> Path:
    """Build a fake fetched folder with real PNGs, optional CSV and a marker."""
    folder = root / "data" / name
    rows = ["image_name,set,subset,identity"]
    for shade, (label, count) in enumerate(labels.items()):
        (folder / label).mkdir(parents=True)
        for index in range(count):
            image_name = f"{label}_{index:02}.png"
            side = 300 if index % 2 == 0 else 256
            color = (index * 9 + 7, 40 + shade * 60, 90)
            with Image.new("RGB", (side, side), color) as image:
                image.save(folder / label / image_name)
            rows.append(f"{image_name},{label},{label},{name}:{label}/{image_name}")
    if with_metadata:
        (folder / "source-metadata.csv").write_text("\n".join(rows) + "\n")
    place_marker(
        folder,
        completed_marker(
            source_name=name,
            declared_splits=with_metadata,
            split_mapping=SPLIT_MAPPING if with_metadata else {},
        ),
    )
    return folder


def request_for(
    tmp_path: Path, folder: Path, **overrides: object
) -> DatasetPreparationRequest:
    """Build the preparation request a fetch would hand to prepare_dataset."""
    values: dict[str, object] = {
        "output_root": str(tmp_path / "datasets"),
        "metadata_file": "source-metadata.csv",
        "split_mapping": SPLIT_MAPPING,
        "dataset_name": folder.name,
    }
    values.update(overrides)
    return remote_request(folder, **values)


def published_records(
    request: DatasetPreparationRequest, revision: str
) -> tuple[Path, list[DatasetImageRecord]]:
    """Locate the published revision and parse its frozen image records."""
    directory = Path(request.output_root) / request.dataset_name / revision
    return directory, read_records(directory, "records.jsonl", DatasetImageRecord)


@pytest.fixture(autouse=True)
def run_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep run reports inside the test folder."""
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))


def test_declared_splits_and_identities_come_from_the_source_metadata(
    tmp_path: Path,
) -> None:
    """Upstream labels map through the request and the marker is frozen."""
    folder = materialized_folder(
        tmp_path, "fake_source", {"validation": 2, "training": 3}, with_metadata=True
    )
    request = request_for(tmp_path, folder)
    summary = prepare_dataset(request)
    assert summary.accepted_count == 5 and summary.full_coverage
    directory, records = published_records(request, summary.revision)
    manifest = load_manifest(directory)
    assert manifest.split_mapping == SPLIT_MAPPING
    assert PROVENANCE_KEYS.issubset(manifest.source_provenance)
    assert manifest.source_provenance["training_intended"] == "true"
    assert manifest.source_provenance["source_name"] == "fake_source"
    assert manifest.metadata_checksum is not None
    assert manifest.expected_by_split == {"validation": 2, "training": 3}
    for record in records:
        label = record.source_path.split("/")[0]
        assert record.upstream_split == label
        assert record.assigned_split == SPLIT_MAPPING[label]
        assert record.source_identity == f"fake_source:{record.source_path}"
        assert record.subset == label
    assert summary.unique_eligible_by_split == {
        "train": 3,
        "tuning": 0,
        "held_out": 2,
    }
    assert validate_dataset(directory).integrity == "verified"
    again = prepare_dataset(request)
    assert again.reused and again.revision == summary.revision


def test_metadata_without_a_request_mapping_uses_the_default_labels(
    tmp_path: Path,
) -> None:
    """The frozen mapping is always the one used to read the source labels."""
    folder = materialized_folder(
        tmp_path, "fake_default", {"validation": 1, "training": 2}, with_metadata=True
    )
    request = request_for(tmp_path, folder, split_mapping=None)
    summary = prepare_dataset(request)
    directory, records = published_records(request, summary.revision)
    assert load_manifest(directory).split_mapping == DEFAULT_SPLIT_MAPPING
    assert {record.assigned_split for record in records} == {"train", "tuning"}
    assert validate_dataset(directory).integrity == "verified"


def test_evaluation_only_sources_prepare_without_training_images(
    tmp_path: Path,
) -> None:
    """A held-out-only source validates when training was never intended."""
    folder = materialized_folder(
        tmp_path, "fake_eval", {"validation": 3}, with_metadata=True
    )
    request = request_for(tmp_path, folder, training_intended=False)
    summary = prepare_dataset(request)
    assert summary.unique_eligible_by_split["train"] == 0
    assert summary.unique_eligible_by_split["held_out"] == 3
    directory, _ = published_records(request, summary.revision)
    manifest = load_manifest(directory)
    assert manifest.source_provenance["training_intended"] == "false"
    assert validate_dataset(directory).integrity == "verified"
    with pytest.raises(ApplicationFailure) as failure:
        prepare_dataset(request_for(tmp_path, folder, training_intended=True))
    assert failure.value.code == "dataset_training_empty"


def test_sources_without_metadata_use_generated_splits(tmp_path: Path) -> None:
    """Without a CSV the mapping is dropped and seeded splits are generated."""
    folder = materialized_folder(
        tmp_path, "fake_plain", {"validation": 6, "training": 6}, with_metadata=False
    )
    request = request_for(tmp_path, folder, metadata_file=None)
    summary = prepare_dataset(request)
    assert summary.accepted_count == 12 and summary.full_coverage
    directory, records = published_records(request, summary.revision)
    manifest = load_manifest(directory)
    assert manifest.split_mapping == {} and manifest.metadata_checksum is None
    assert all(
        record.upstream_split is None and record.source_identity is None
        for record in records
    )
    assert summary.unique_eligible_by_split["train"] > 0
    assert validate_dataset(directory).integrity == "verified"


def test_remote_folders_need_their_marker_before_preparation(tmp_path: Path) -> None:
    """A folder that lost its marker cannot be prepared as a remote source."""
    folder = materialized_folder(
        tmp_path, "fake_lost", {"training": 2}, with_metadata=True
    )
    (folder / ".stegolab_source.json").unlink()
    with pytest.raises(ApplicationFailure) as failure:
        prepare_dataset(request_for(tmp_path, folder))
    assert failure.value.code == "source_marker_missing"
    assert not list((tmp_path / "datasets").glob("fake_lost/*"))
