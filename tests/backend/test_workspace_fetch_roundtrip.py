"""Upload a zip, fetch it through a real worker and see the result in the API."""

import time
from pathlib import Path

import pytest
from archive_fixtures import ZipMember, build_zip
from fastapi.testclient import TestClient
from fetch_fixtures import smooth_png

from backend_service.application import create_application
from schemas.jobs import JobSnapshot

PREFIX = "/api/v1"
UPLOADS = f"{PREFIX}/datasets/uploads"
FETCH_JOBS = f"{PREFIX}/datasets/fetch_jobs"
OCTET_STREAM = {"Content-Type": "application/octet-stream"}
WAIT_SECONDS = 120
ARCHIVE = build_zip(
    [ZipMember(f"images/{number}.png", smooth_png(number)) for number in (1, 2)]
)


def upload_archive(client: TestClient) -> str:
    """Send the small archive as one part and complete the upload session."""
    created = client.post(
        UPLOADS,
        json={
            "client_request_identifier": "roundtrip-upload",
            "file_name": "browser.zip",
            "total_bytes": len(ARCHIVE),
        },
    )
    assert created.status_code == 201, created.text
    identifier = str(created.json()["upload_identifier"])
    part = client.put(
        f"{UPLOADS}/{identifier}/chunks/0", content=ARCHIVE, headers=OCTET_STREAM
    )
    assert part.status_code == 200, part.text
    finished = client.post(
        f"{UPLOADS}/{identifier}/complete",
        json={"client_request_identifier": "roundtrip-upload-complete"},
    )
    assert finished.status_code == 200, finished.text
    assert finished.json()["complete"] is True
    return identifier


def wait_for_job(client: TestClient, identifier: str) -> JobSnapshot:
    """Wait a bounded time for the real worker to publish its final record."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        response = client.get(f"{PREFIX}/jobs/{identifier}")
        assert response.status_code == 200, response.text
        snapshot = JobSnapshot.model_validate_json(response.text)
        if snapshot.status not in ("queued", "running"):
            return snapshot
        time.sleep(0.1)
    raise AssertionError("The fetch worker did not finish within the wait limit.")


def test_uploaded_archive_becomes_a_raw_folder_and_a_prepared_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One real subprocess fetch: upload, queue, poll, then read the workspace."""
    monkeypatch.delenv("STEGOLAB_CACHE_DIRECTORY", raising=False)
    root = tmp_path / "workspace"
    # The deployment mounts data/ before the API starts, so the catalog lists it.
    (root / "data").mkdir(parents=True)
    application = create_application(tmp_path / "state", tmp_path / "logs", root)
    with TestClient(application, raise_server_exceptions=False) as client:
        upload = upload_archive(client)
        accepted = client.post(
            FETCH_JOBS,
            json={
                "client_request_identifier": "roundtrip-fetch",
                "source": {"source_kind": "upload", "upload_identifier": upload},
                "source_name": "browser_zip",
                "dataset_name": "browser_prepared",
                "training_intended": False,
            },
        )
        assert accepted.status_code == 202, accepted.text
        queued = JobSnapshot.model_validate_json(accepted.text)
        assert (queued.operation, queued.available_actions) == (
            "fetch_dataset",
            ["cancel"],
        )
        final = wait_for_job(client, queued.job_identifier)
        assert final.status == "completed", final.error
        assert (final.phase, final.progress, final.available_actions) == (
            "completed",
            1.0,
            [],
        )
        assert final.result is not None
        assert final.result["raw_folder"] == "data/browser_zip"
        dataset = final.result["dataset"]
        assert isinstance(dataset, dict) and dataset["accepted_count"] == 2
        workspace = client.get(f"{PREFIX}/workspace").json()
        assert any("browser_zip" in item["folders"] for item in workspace["sources"])
        prepared = [
            item for item in workspace["datasets"] if item["name"] == "browser_prepared"
        ]
        assert len(prepared) == 1 and prepared[0]["image_count"] == 2
        assert prepared[0]["revision"] == dataset["revision"]
    assert (root / "data" / "browser_zip" / "images" / "1.png").is_file()
    run_label = queued.job_identifier.removeprefix("job_")
    run_folders = sorted((root / "logs").glob(f"*_{run_label}_*"))
    assert len(run_folders) == 1
    names = {path.name for path in run_folders[0].iterdir()}
    assert {"events.jsonl", "fetch_progress.json", "dataset_fetch_run.json"} <= names
