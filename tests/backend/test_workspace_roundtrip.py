"""Exercise the browser API through a real isolated training subprocess."""

import hashlib
import time
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend_service.application import create_application
from backend_service.dataset_image import prepare_dataset_image
from backend_service.dataset_manifest import write_manifest
from schemas.datasets import DatasetImageRecord, DatasetPreparationRequest
from schemas.jobs import JobSnapshot


def prepare_sample(root: Path) -> None:
    """Publish eight synthetic covers with explicit train and tuning identities."""
    stage = root / "datasets" / "browser_fixture" / "stage"
    (stage / "images").mkdir(parents=True)
    records = []
    for number in range(1, 9):
        stream = BytesIO()
        with Image.new("RGB", (1024, 1024), (number, number * 2, number * 3)) as cover:
            cover.save(stream, format="PNG")
        content = stream.getvalue()
        relative = f"images/{number}.png"
        prepared = prepare_dataset_image(content, stage / relative, lambda count: None)
        records.append(
            DatasetImageRecord(
                source_path=f"{number}.png",
                prepared_path=relative,
                source_identity=f"fixture:{number}",
                upstream_split="training" if number <= 4 else "validation",
                source_checksum=hashlib.sha256(content).hexdigest(),
                source_bytes=len(content),
                **asdict(prepared),
            )
        )
    manifest = write_manifest(
        stage,
        DatasetPreparationRequest(
            source_directory="unused", dataset_name="browser_fixture"
        ),
        records,
        [],
        before_write=lambda count: None,
        metadata_checksum="a" * 64,
        expected_images=8,
    )
    stage.rename(stage.parent / manifest.revision)


def wait_for_job(client: TestClient, identifier: str) -> JobSnapshot:
    """Wait a bounded time for a real subprocess to publish its final record."""
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{identifier}")
        assert response.status_code == 200
        snapshot = JobSnapshot.model_validate_json(response.text)
        if snapshot.status not in ("queued", "running"):
            return snapshot
        time.sleep(0.1)
    raise AssertionError("The isolated worker did not finish within 45 seconds.")


def test_real_api_train_reload_and_resume_share_one_allowance(tmp_path: Path) -> None:
    """Run real updates through API workers without using the user's experiment."""
    root = tmp_path / "workspace"
    prepare_sample(root)
    application = create_application(tmp_path / "state", tmp_path / "logs", root)
    with TestClient(application) as client:
        workspace = client.get("/api/v1/workspace").json()
        dataset = workspace["datasets"][0]
        assert dataset["compatible"] is True
        document = {
            "client_request_identifier": "real-browser-first",
            "operation": "train",
            "experiment_identifier": "browser_fixture",
            "dataset_identifier": dataset["identifier"],
            "cpu_threads": 1,
            "stop_after_step": 1,
        }
        preflight = client.post("/api/v1/training_preflight", json=document)
        assert preflight.status_code == 200
        assert preflight.json()["allowed"], preflight.json()["blockers"]
        first = client.post("/api/v1/training_jobs", json=document)
        assert first.status_code in (200, 201, 202), first.text
        identifier = first.json()["job_identifier"]
        retried = client.post("/api/v1/training_jobs", json=document)
        assert retried.json()["job_identifier"] == identifier
        snapshot = wait_for_job(client, identifier)
        assert snapshot.status in ("stopped", "completed"), snapshot.error
        assert snapshot.result is not None
        assert snapshot.result["global_step"] == 1
        checkpoints = client.get("/api/v1/workspace").json()["checkpoints"]
        assert len(checkpoints) == 1
        assert checkpoints[0]["resume_blockers"] == []
        checkpoint = checkpoints[0]["identifier"]

    # A new API lifespan restores metadata without starting learning work again.
    with TestClient(
        create_application(tmp_path / "state", tmp_path / "logs", root)
    ) as client:
        assert (
            client.get(f"/api/v1/jobs/{identifier}").json()["status"] == snapshot.status
        )
        resumed = {
            **document,
            "client_request_identifier": "real-browser-resume",
            "checkpoint_identifier": checkpoint,
            "stop_after_step": 2,
        }
        assert client.post("/api/v1/training_preflight", json=resumed).json()["allowed"]
        response = client.post("/api/v1/training_jobs", json=resumed)
        assert response.status_code in (200, 201, 202), response.text
        final = wait_for_job(client, response.json()["job_identifier"])
        assert final.status in ("stopped", "completed"), final.error
        assert final.result is not None and final.result["global_step"] == 2
        budget = client.get("/api/v1/workspace").json()["budget"]
        assert budget["remaining_experiments"] == 1
        assert 0 < budget["remaining_seconds"] < 14400


def test_real_api_prepares_local_folder_without_spending_budget(tmp_path: Path) -> None:
    """Prepare actual mounted-source images through a worker and block escapes."""
    root = tmp_path / "workspace"
    source = root / "data" / "covers"
    source.mkdir(parents=True)
    for number in range(3):
        with Image.new("RGB", (256, 256), (number + 1, 12, 24)) as cover:
            cover.save(source / f"cover_{number}.png")
    originals = {path.name: path.read_bytes() for path in source.iterdir()}
    with TestClient(
        create_application(tmp_path / "state", tmp_path / "logs", root)
    ) as client:
        workspace = client.get("/api/v1/workspace").json()
        document = {
            "client_request_identifier": "prepare-browser-fixture",
            "operation": "prepare_dataset",
            "source_identifier": workspace["sources"][0]["identifier"],
            "source_subdirectory": "covers",
            "dataset_name": "prepared_fixture",
        }
        escaped = client.post(
            "/api/v1/dataset_jobs",
            json={**document, "source_subdirectory": "../../"},
        )
        assert escaped.status_code == 422
        response = client.post("/api/v1/dataset_jobs", json=document)
        assert response.status_code == 202, response.text
        job = wait_for_job(client, response.json()["job_identifier"])
        assert job.status == "completed", job.error
        workspace = client.get("/api/v1/workspace").json()
        assert workspace["datasets"][0]["image_count"] == 3
        assert not workspace["datasets"][0]["compatible"]
        assert workspace["budget"]["remaining_experiments"] == 2
        assert workspace["budget"]["remaining_seconds"] == 14400
        assert not (root / "state" / "cpu_proof" / "ledger.json").exists()
        assert {path.name: path.read_bytes() for path in source.iterdir()} == originals
