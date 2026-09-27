"""Shared application, upload, marker and cache helpers for dataset route tests."""

from pathlib import Path
from typing import Any

import pytest
from archive_fixtures import ZipMember, build_zip, tiny_png
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_service import workflow_supervision
from backend_service.application import create_application
from backend_service.dataset_serialization import canonical_json
from backend_service.dataset_sources.plans import PlannedAsset
from backend_service.dataset_sources.source_marker import MARKER_NAME
from schemas.dataset_sources import DatasetInspection, SourceMarker

PREFIX = "/api/v1"
INSPECTIONS = f"{PREFIX}/datasets/inspections"
FETCH_JOBS = f"{PREFIX}/datasets/fetch_jobs"
STORAGE = f"{PREFIX}/datasets/storage"
CLEANUP = f"{PREFIX}/datasets/cache/unused"
UPLOADS = f"{PREFIX}/datasets/uploads"
OCTET_STREAM = {"Content-Type": "application/octet-stream"}
SOURCE_KINDS = ["server_folder", "upload", "hugging_face", "https_archive"]
HUB_SOURCE = {
    "source_kind": "hugging_face",
    "repository": "example/tiny",
    "terms_reference": "https://huggingface.co/datasets/example/tiny",
}
ARCHIVE = build_zip(
    [ZipMember(f"images/{number}.png", tiny_png((number, 0, 0))) for number in (1, 2)]
)
ASSET_SHA256 = "c" * 64
CREATED_AT = "2026-09-27T00:00:00+00:00"
MARKER_REFERENCE = "https://example.org/fake.zip"
MARKER_REVISION = "etag-1"
MARKER_FIELDS: dict[str, object] = {
    "source_kind": "https_archive",
    "reference": MARKER_REFERENCE,
    "source_url": MARKER_REFERENCE,
    "terms_reference": "https://example.org/terms",
    "resolved_revision": MARKER_REVISION,
    "materialization_identity": "b" * 64,
    "content": "images",
    "assets": [{"path": "fake.zip", "size_bytes": 1234, "sha256": ASSET_SHA256}],
    "member_count": 2,
    "rejected_member_count": 0,
    "declared_splits": False,
    "created_at": CREATED_AT,
    "completed": True,
}


def build_application(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Build an API on isolated folders whose scheduler never starts a job."""
    monkeypatch.setattr(workflow_supervision, "next_job", lambda service: None)
    return create_application(tmp_path / "data", tmp_path / "logs", tmp_path / "root")


def fake_inspection(access: str = "available") -> DatasetInspection:
    """Build the inspection a fake source layer answers with."""
    return DatasetInspection.model_validate(
        {
            "source_kind": "hugging_face",
            "reference": "https://huggingface.co/datasets/example/tiny",
            "suggested_source_name": "tiny",
            "access": access,
            "access_guidance": None if access == "available" else "Accept the terms.",
            "content": "images",
            "asset_count": 1,
            "supports_pause": True,
            "declared_splits": False,
            "materialization_identity": "a" * 64,
            "free_disk_bytes": 10**12,
            "server_token_configured": False,
        }
    )


def marked_folder(data_root: Path, name: str) -> SourceMarker:
    """Create a raw source folder with one asset file and a completed marker."""
    marker = SourceMarker.model_validate({**MARKER_FIELDS, "source_name": name})
    folder = data_root / name
    folder.mkdir(parents=True)
    (folder / "fake.zip").write_bytes(b"z" * 1234)
    (folder / MARKER_NAME).write_bytes(canonical_json(marker.model_dump(mode="json")))
    return marker


def planned_asset(path: str, size: int) -> PlannedAsset:
    """Describe one local file the cache can import or download."""
    return PlannedAsset(None, path, size, None, None, False, None)


def failure(response: Any) -> tuple[int, str]:
    """Read the status and fixed error code of a failed response."""
    return response.status_code, str(response.json()["error"]["code"])


def upload_archive(client: TestClient, request_identifier: str, complete: bool) -> str:
    """Upload the small zip through the routes, optionally leaving it incomplete."""
    body = {
        "client_request_identifier": request_identifier,
        "file_name": "tiny.zip",
        "total_bytes": len(ARCHIVE),
    }
    created = client.post(UPLOADS, json=body)
    assert created.status_code == 201
    identifier = str(created.json()["upload_identifier"])
    if complete:
        part = client.put(
            f"{UPLOADS}/{identifier}/chunks/0", content=ARCHIVE, headers=OCTET_STREAM
        )
        finished = client.post(
            f"{UPLOADS}/{identifier}/complete",
            json={"client_request_identifier": f"{request_identifier}-complete"},
        )
        assert (part.status_code, finished.status_code) == (200, 200)
    return identifier


def fetch_body(upload: str, identifier: str = "fetch-1", **overrides: Any) -> Any:
    """Build a fetch request for an uploaded archive."""
    return {
        "client_request_identifier": identifier,
        "source": {"source_kind": "upload", "upload_identifier": upload},
        "source_name": "browser_zip",
        "dataset_name": "browser_prepared",
        "training_intended": False,
        **overrides,
    }
