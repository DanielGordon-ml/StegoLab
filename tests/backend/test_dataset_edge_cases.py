"""Verify public dataset preparation boundaries using disposable source files."""

import errno
import hashlib
import json
import shutil
from collections.abc import Buffer
from pathlib import Path

import pytest
from PIL import Image

from backend_service.dataset_preparation import prepare_dataset
from backend_service.dataset_serialization import read_records
from backend_service.dataset_validation import validate_dataset
from backend_service.failures import ApplicationFailure
from backend_service.image_output import BoundedPngWriter
from schemas.datasets import (
    DatasetImageRecord,
    DatasetPreparationRequest,
    DatasetRejection,
)


@pytest.fixture
def edge_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> DatasetPreparationRequest:
    """Create one distinct source in each official split with local run logs."""
    source = tmp_path / "sources"
    rows = ["image_name,set,subset"]
    for index, split in enumerate(("training", "validation", "test"), 1):
        directory = source / split
        directory.mkdir(parents=True)
        Image.new("RGB", (256, 257), (index, 30, 50)).save(directory / f"{index}.png")
        rows.append(f"{index}.png,{split},example")
    (source / "uhd-iqa-metadata.csv").write_text("\n".join(rows) + "\n")
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))
    return DatasetPreparationRequest(
        source_directory=str(source), output_root=str(tmp_path / "datasets")
    )


def assert_no_publication(request: DatasetPreparationRequest) -> None:
    """Require failed imports to clean their own stage and publish no revision."""
    root = Path(request.output_root)
    assert not list(root.glob("uhd_iqa/*"))
    assert not list((root / ".staging").iterdir())


def test_all_ineligible_input_fails_without_publication(
    edge_request: DatasetPreparationRequest,
) -> None:
    """Valid tiny images stay ineligible and never become a usable training set."""
    for index, path in enumerate(Path(edge_request.source_directory).rglob("*.png")):
        Image.new("RGB", (255, 128), (index, 41, 21)).save(path)
    with pytest.raises(ApplicationFailure, match="at least 256 pixels"):
        prepare_dataset(edge_request)
    assert_no_publication(edge_request)


def test_named_partial_selection_preserves_rejected_upstream_coverage(
    edge_request: DatasetPreparationRequest,
) -> None:
    """Named subsets keep full expected counts and include invalid selected bytes."""
    source = Path(edge_request.source_directory)
    (source / "validation/2.png").write_bytes(b"invalid selected PNG")
    request = edge_request.model_copy(
        update={
            "selection_name": "focused_sample",
            "selection": ["validation/2.png", "training/1.png"],
        }
    )
    result = prepare_dataset(request)
    revision = Path(request.output_root) / request.dataset_name / result.revision
    assert result.accepted_count == 1 and result.rejection_count == 1
    assert result.expected_images == 3 and result.discovered_images == 3
    assert result.selected_images == 2 and not result.full_coverage
    rejections = read_records(revision, "rejections.jsonl", DatasetRejection)
    assert rejections[0].upstream_split == "validation"
    assert (
        rejections[0].source_checksum
        == hashlib.sha256(b"invalid selected PNG").hexdigest()
    )
    manifest = json.loads((revision / "manifest.json").read_bytes())
    assert manifest["selected_by_split"] == {"training": 1, "validation": 1}
    assert manifest["selection"] == ["training/1.png", "validation/2.png"]
    (source / "validation/2.png").write_bytes(b"different invalid selected PNG")
    changed = prepare_dataset(request)
    assert changed.revision != result.revision
    validate_dataset(revision)


@pytest.mark.parametrize("failure", ["wrong_directory", "unknown_split", "missing_key"])
def test_invalid_metadata_mapping_blocks_import(
    edge_request: DatasetPreparationRequest, failure: str
) -> None:
    """Source layout, unknown split labels, and missing selected identities fail."""
    source = Path(edge_request.source_directory)
    metadata = source / "uhd-iqa-metadata.csv"
    if failure == "wrong_directory":
        (source / "training/1.png").rename(source / "validation/1.png")
    elif failure == "unknown_split":
        metadata.write_text(
            metadata.read_text().replace(",training,", ",experimental,")
        )
    else:
        metadata.write_text(
            metadata.read_text().replace("1.png,training,example\n", "")
        )
        edge_request = edge_request.model_copy(
            update={
                "selection_name": "unknown_identity",
                "selection": ["training/1.png"],
            }
        )
    with pytest.raises(ApplicationFailure, match="metadata"):
        prepare_dataset(edge_request)
    assert_no_publication(edge_request)


def test_explicit_subset_allows_missing_unselected_source_files(
    edge_request: DatasetPreparationRequest,
) -> None:
    """An explicitly named local subset never claims complete upstream coverage."""
    source = Path(edge_request.source_directory)
    (source / "test/3.png").unlink()
    request = edge_request.model_copy(
        update={
            "selection_name": "available_train",
            "selection": ["training/1.png"],
        }
    )
    result = prepare_dataset(request)
    assert (
        result.expected_images,
        result.discovered_images,
        result.selected_images,
    ) == (3, 2, 1)
    assert not result.full_coverage
    with pytest.raises(ApplicationFailure, match="metadata"):
        prepare_dataset(edge_request)


def test_local_generated_splits_survive_relocation_and_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlabelled local data keeps frozen seed-based groups across source roots."""
    source = tmp_path / "local_source"
    source.mkdir()
    for index in range(24):
        Image.new("RGB", (256, 256), (index, 12, 33)).save(source / f"{index:02}.png")
    request = DatasetPreparationRequest(
        source_directory=str(source),
        source_kind="local",
        metadata_file=None,
        dataset_name="development",
        output_root=str(tmp_path / "prepared"),
        seed=23,
    )
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))
    first = prepare_dataset(request)
    moved = tmp_path / "moved"
    shutil.copytree(source, moved)
    second = prepare_dataset(
        request.model_copy(update={"source_directory": str(moved)})
    )
    assert second.revision == first.revision and second.reused
    path = Path(request.output_root) / request.dataset_name / first.revision
    records = read_records(path, "records.jsonl", DatasetImageRecord)
    assert all(item.upstream_split is None for item in records)
    assert set(first.split_counts) == {"train", "tuning", "held_out"}
    assert first.unique_eligible_by_split["train"] > 0


def test_disk_full_during_image_write_preserves_existing_revision(
    edge_request: DatasetPreparationRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An actual encoded-write OSError removes staging and leaves prior data valid."""
    first = prepare_dataset(edge_request)
    source = Path(edge_request.source_directory)
    Image.new("RGB", (256, 257), (90, 91, 92)).save(source / "training/1.png")

    def disk_full(self: BoundedPngWriter, buffer: Buffer) -> int:
        """Inject disk exhaustion during an output write."""
        self.stream.write(memoryview(buffer)[:1])
        raise OSError(errno.ENOSPC, "simulated full filesystem")

    with monkeypatch.context() as temporary:
        temporary.setattr(BoundedPngWriter, "write", disk_full)
        with pytest.raises(ApplicationFailure):
            prepare_dataset(edge_request)
    root = Path(edge_request.output_root)
    assert not list((root / ".staging").iterdir())
    revisions = list((root / edge_request.dataset_name).iterdir())
    assert [path.name for path in revisions] == [first.revision]
    assert validate_dataset(revisions[0]).integrity == "verified"


@pytest.mark.parametrize(
    "source_kind,references,expected",
    [
        ("local", {}, ("local", "not_reviewed")),
        (
            "local",
            {"source_url": "owner_collection"},
            ("owner_collection", "not_reviewed"),
        ),
        (
            "local",
            {"source_url": "owner_collection", "terms_reference": "owner_terms"},
            ("owner_collection", "owner_terms"),
        ),
        (
            "uhd_iqa",
            {},
            (
                "https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html",
                "https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html",
            ),
        ),
    ],
)
def test_source_provenance_defaults_match_source_kind(
    source_kind: str, references: dict[str, str], expected: tuple[str, str]
) -> None:
    """Local defaults stay truthful and caller-supplied references are preserved."""
    request = DatasetPreparationRequest.model_validate(
        {
            "source_directory": "sources",
            "source_kind": source_kind,
            **references,
        }
    )
    assert (request.source_url, request.terms_reference) == expected
    revalidated = DatasetPreparationRequest.model_validate(request)
    assert (revalidated.source_url, revalidated.terms_reference) == expected
