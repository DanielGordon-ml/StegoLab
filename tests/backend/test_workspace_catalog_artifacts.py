"""Registered artifact downloads reject changes and package traversal entries."""

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from backend_service.failures import ApplicationFailure
from backend_service.workspace_catalog import WorkspaceCatalog
from schemas.model_exports import ExportFile, ModelExportManifest


def package(root: Path) -> Path:
    """Write public non-executable package bytes for archive-boundary tests."""
    directory = root / "models" / "example" / "encoder"
    directory.mkdir(parents=True)
    files: dict[str, ExportFile] = {}
    for name in ("model.pt2", "runtime.py"):
        content = b"public package fixture"
        (directory / name).write_bytes(content)
        files[name] = ExportFile(
            checksum=hashlib.sha256(content).hexdigest(), size_bytes=len(content)
        )
    manifest = ModelExportManifest(
        compatibility_identifier="dense_v1_" + "a" * 64,
        source_identifier="public_fixture",
        role="encoder",
        dependencies={"torch": "fixture"},
        producer_environment={"system": "fixture"},
        files=files,
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json())
    return directory


def test_only_registered_verified_export_members_are_downloadable(
    tmp_path: Path,
) -> None:
    """ZIP output contains exact declared files and never unrelated private files."""
    directory = package(tmp_path)
    (directory / "private.txt").write_text("not part of the package")
    catalog = WorkspaceCatalog(tmp_path, tmp_path / "private", source_roots={})
    identifier = catalog.snapshot().exports[0].artifact_identifier
    content, name = catalog.download_export(identifier)
    assert name == "stegolab-encoder-experimental.zip"
    with ZipFile(BytesIO(content)) as archive:
        assert set(archive.namelist()) == {"manifest.json", "model.pt2", "runtime.py"}
        assert archive.read("runtime.py") == (directory / "runtime.py").read_bytes()
    (directory / "runtime.py").write_bytes(b"changed")
    with pytest.raises(ApplicationFailure, match="missing or changed"):
        catalog.download_export(identifier)
    with pytest.raises(ApplicationFailure):
        catalog.download_export(str(directory))


@pytest.mark.parametrize(
    "entry", ["../secret.txt", "/secret.txt", "a/../../secret.txt"]
)
def test_package_manifest_cannot_escape_export_root(tmp_path: Path, entry: str) -> None:
    """Untrusted archive member names never become filesystem or ZIP traversal."""
    directory = package(tmp_path)
    path = directory / "manifest.json"
    manifest = ModelExportManifest.model_validate_json(path.read_bytes())
    manifest.files[entry] = next(iter(manifest.files.values()))
    path.write_text(manifest.model_dump_json())
    catalog = WorkspaceCatalog(tmp_path, tmp_path / "private", source_roots={})
    identifier = catalog.snapshot().exports[0].artifact_identifier
    with pytest.raises(ApplicationFailure):
        catalog.download_export(identifier)


def test_package_symlinks_are_rejected_after_registration(tmp_path: Path) -> None:
    """A later artifact replacement cannot redirect a saved opaque reference."""
    directory = package(tmp_path)
    catalog = WorkspaceCatalog(tmp_path, tmp_path / "private", source_roots={})
    identifier = catalog.snapshot().exports[0].artifact_identifier
    original = directory / "runtime.py"
    target = tmp_path / "secret.txt"
    target.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(target)
    with pytest.raises(ApplicationFailure):
        catalog.download_export(identifier)
