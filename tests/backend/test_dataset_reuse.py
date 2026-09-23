"""Check early reuse against current bytes before any new image staging."""

from pathlib import Path

import pytest
from PIL import Image

from backend_service.dataset_inventory import inventory_dataset
from backend_service.dataset_preparation import prepare_dataset
from backend_service.dataset_reuse import find_reusable_dataset
from backend_service.dataset_validation import load_manifest
from backend_service.failures import ApplicationFailure
from schemas.datasets import DatasetPreparationRequest


@pytest.fixture
def prepared_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DatasetPreparationRequest, Path, dict[str, str]]:
    """Publish one valid training cover with source metadata and an ignored file."""
    source = tmp_path / "source"
    source.mkdir()
    with Image.new("RGB", (256, 257), (39, 71, 113)) as image:
        image.save(source / "1.png")
    (source / "uhd-iqa-metadata.csv").write_text(
        "image_name,set,subset\n1.png,training,\n"
    )
    request = DatasetPreparationRequest(
        source_directory=str(source), output_root=str(tmp_path / "output")
    )
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))
    summary = prepare_dataset(request)
    directory = Path(request.output_root) / request.dataset_name / summary.revision
    return request, directory, load_manifest(directory).source_provenance


def test_matching_input_reuses_without_preparing_images(
    prepared_request: tuple[DatasetPreparationRequest, Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate existing pixels while forbidding another encode/write pass."""
    request, directory, provenance = prepared_request

    def forbidden(*arguments: object, **keywords: object) -> None:
        """Fail if reuse accidentally starts a preparation operation."""
        pytest.fail("Reuse must not prepare new images.")

    monkeypatch.setattr(
        "backend_service.dataset_image.prepare_dataset_image", forbidden
    )
    result = find_reusable_dataset(request, inventory_dataset(request), provenance)
    assert result is not None
    assert result.reused and result.integrity == "verified"
    assert result.revision == directory.name


def test_changed_source_or_tooling_disables_reuse(
    prepared_request: tuple[DatasetPreparationRequest, Path, dict[str, str]],
) -> None:
    """Require source bytes and supported preparation tooling to match exactly."""
    request, _directory, provenance = prepared_request
    inventory = inventory_dataset(request)
    changed = provenance | {"preparation": "different-version"}
    assert find_reusable_dataset(request, inventory, changed) is None
    with Image.new("RGB", (256, 257), (53, 97, 127)) as image:
        image.save(Path(request.source_directory) / "1.png")
    assert (
        find_reusable_dataset(request, inventory_dataset(request), provenance) is None
    )


def test_missing_current_checksum_disables_reuse(
    prepared_request: tuple[DatasetPreparationRequest, Path, dict[str, str]],
) -> None:
    """Avoid reuse when any selected source identity has not been verified."""
    request, _directory, provenance = prepared_request
    inventory = inventory_dataset(request)
    inventory.source_checksums.clear()
    assert find_reusable_dataset(request, inventory, provenance) is None


def test_corrupt_matching_pixels_fail_integrity(
    prepared_request: tuple[DatasetPreparationRequest, Path, dict[str, str]],
) -> None:
    """Never hide a corrupted matching completed revision behind a new import."""
    request, directory, provenance = prepared_request
    prepared = next((directory / "images").iterdir())
    with prepared.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ApplicationFailure) as failure:
        find_reusable_dataset(request, inventory_dataset(request), provenance)
    assert failure.value.code == "dataset_integrity"


def test_candidate_symlink_is_rejected(
    prepared_request: tuple[DatasetPreparationRequest, Path, dict[str, str]],
) -> None:
    """Refuse revision entries that could redirect scanning outside the output."""
    request, directory, provenance = prepared_request
    (directory.parent / ("a" * 64)).symlink_to(directory, target_is_directory=True)
    with pytest.raises(ApplicationFailure) as failure:
        find_reusable_dataset(request, inventory_dataset(request), provenance)
    assert failure.value.code == "dataset_integrity"
