"""Exercise the public dataset workflow, repeatability, and safe failures."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from backend_service.command_line import main
from backend_service.dataset_commands import DatasetTerminated
from backend_service.dataset_preparation import prepare_dataset
from backend_service.dataset_validation import validate_dataset
from backend_service.failures import ApplicationFailure
from schemas.datasets import DatasetPreparationRequest


@pytest.fixture
def dataset_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create distinct RGB and grayscale covers in three official source splits."""
    source = tmp_path / "source"
    rows = ["image_name,set,subset"]
    for number, split in enumerate(("training", "validation", "test"), 1):
        directory = source / split
        directory.mkdir(parents=True)
        mode = "L" if number == 2 else "RGB"
        color = 47 if mode == "L" else (number * 31, 53, 90)
        with Image.new(mode, (256, 257), color) as image:
            image.save(directory / f"{number}.png")
        rows.append(f"{number}.png,{split},")
    (source / "uhd-iqa-metadata.csv").write_text("\n".join(rows) + "\n")
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {"source_directory": str(source), "output_root": str(tmp_path / "prepared")}
        )
    )
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))
    return path


def test_cli_prepare_inspect_validate_and_reuse(
    dataset_request: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run the real command flow, preserving source files and the same revision."""
    request = DatasetPreparationRequest.model_validate_json(dataset_request.read_text())
    source = Path(request.source_directory)
    original = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*.png")
    }
    assert main(["prepare_dataset", str(dataset_request)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["accepted_count"] == 3
    assert first["full_coverage"] is True
    revision = Path(request.output_root) / "uhd_iqa" / first["revision"]
    assert main(["inspect_dataset", str(revision)]) == 0
    assert json.loads(capsys.readouterr().out)["integrity"] == "not_checked"
    assert main(["prepare_dataset", str(dataset_request)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["revision"] == first["revision"] and second["reused"]
    assert original == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*.png")
    }
    source.rename(source.with_name("unmounted"))
    assert main(["validate_dataset", str(revision)]) == 0
    assert json.loads(capsys.readouterr().out)["integrity"] == "verified"
    prepared = next((revision / "images").glob("*.png"))
    prepared.write_bytes(b"damaged")
    assert main(["validate_dataset", str(revision)]) == 1
    assert "integrity" in capsys.readouterr().err


def test_relocated_source_and_changed_input(dataset_request: Path) -> None:
    """Absolute roots never change revision identity; changed pixels always do."""
    request = DatasetPreparationRequest.model_validate_json(dataset_request.read_text())
    first = prepare_dataset(request)
    moved = dataset_request.parent / "moved"
    shutil.copytree(request.source_directory, moved)
    relocated = request.model_copy(update={"source_directory": str(moved)})
    assert prepare_dataset(relocated).revision == first.revision
    with Image.new("RGB", (256, 257), (210, 12, 41)) as image:
        image.save(moved / "training" / "1.png")
    assert prepare_dataset(relocated).revision != first.revision


def test_invalid_file_is_reported_without_false_full_coverage(
    dataset_request: Path,
) -> None:
    """Unknown files and invalid image bytes appear as explicit rejection counts."""
    request = DatasetPreparationRequest.model_validate_json(dataset_request.read_text())
    (Path(request.source_directory) / "notes.txt").write_text("not an image")
    result = prepare_dataset(request)
    assert result.rejection_count == 1
    assert result.rejection_reasons == {"unsupported_file_type": 1}
    assert not result.full_coverage


def test_conflicting_duplicates_never_publish(dataset_request: Path) -> None:
    """Identical visible pixels declared in separate official splits must fail."""
    request = DatasetPreparationRequest.model_validate_json(dataset_request.read_text())
    source = Path(request.source_directory)
    shutil.copyfile(source / "training/1.png", source / "test/3.png")
    with pytest.raises(ApplicationFailure, match="split or duplicate"):
        prepare_dataset(request)
    assert not list(Path(request.output_root).glob("uhd_iqa/*"))
    assert not list((Path(request.output_root) / ".staging").iterdir())


def test_invalid_request_returns_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid JSON fields fail safely without echoing untrusted content."""
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))
    request = tmp_path / "bad.json"
    request.write_text('{"source_directory":"PRIVATE", "unknown":true}')
    assert main(["prepare_dataset", str(request)]) == 2
    assert "PRIVATE" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "interruption,exit_code", [(KeyboardInterrupt, 130), (DatasetTerminated, 143)]
)
def test_cli_interruption_cleans_staging(
    dataset_request: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
    exit_code: int,
) -> None:
    """Both supported interruption paths release the lock and remove the stage."""

    def interrupt(*arguments: object, **keywords: object) -> None:
        """Inject an interruption while the import owns its staging directory."""
        raise interruption

    monkeypatch.setattr(
        "backend_service.dataset_import.prepare_dataset_image", interrupt
    )
    assert main(["prepare_dataset", str(dataset_request)]) == exit_code
    request = DatasetPreparationRequest.model_validate_json(dataset_request.read_text())
    assert not list((Path(request.output_root) / ".staging").iterdir())


def test_missing_prepared_file_fails(dataset_request: Path) -> None:
    """An otherwise valid manifest cannot hide a deleted prepared image."""
    request = DatasetPreparationRequest.model_validate_json(dataset_request.read_text())
    result = prepare_dataset(request)
    revision = Path(request.output_root) / "uhd_iqa" / result.revision
    next((revision / "images").glob("*.png")).unlink()
    with pytest.raises(ApplicationFailure):
        validate_dataset(revision)


def test_empty_local_source_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty folder is reported as missing input rather than a size violation."""
    source = tmp_path / "empty"
    source.mkdir()
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))
    request = DatasetPreparationRequest(
        source_directory=str(source),
        output_root=str(tmp_path / "prepared"),
        source_kind="local",
        metadata_file=None,
    )
    with pytest.raises(ApplicationFailure, match="no image candidates"):
        prepare_dataset(request)
