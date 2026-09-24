"""Verified ZIP downloads of explicitly registered independent model packages."""

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from backend_service.dataset_serialization import read_bounded
from backend_service.failures import ApplicationFailure
from backend_service.workspace_registry import safe_path
from schemas.model_exports import ModelExportManifest


def export_archive(directory: Path) -> tuple[bytes, str]:
    """Archive declared verified files only, without loading executable graphs."""
    try:
        manifest_bytes = read_bounded(directory / "manifest.json", 256 * 1024)
        manifest = ModelExportManifest.model_validate_json(manifest_bytes)
        if "model.pt2" not in manifest.files or "runtime.py" not in manifest.files:
            raise ValueError
        output = BytesIO()
        total = len(manifest_bytes)
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", manifest_bytes)
            for relative, entry in manifest.files.items():
                if (
                    Path(relative).is_absolute()
                    or "\\" in relative
                    or any(part in ("", ".", "..") for part in relative.split("/"))
                ):
                    raise ValueError
                content = read_bounded(
                    safe_path(directory / relative, directory), 16 * 1024**2
                )
                total += len(content)
                if (
                    total > 64 * 1024**2
                    or len(content) != entry.size_bytes
                    or hashlib.sha256(content).hexdigest() != entry.checksum
                ):
                    raise ValueError
                archive.writestr(relative, content)
        return output.getvalue(), f"stegolab-{manifest.role}-experimental.zip"
    except (OSError, ValueError, ApplicationFailure):
        raise ApplicationFailure(
            "workspace_artifact_unavailable",
            "The package is missing or changed. Create a new verified export.",
            422,
        ) from None
