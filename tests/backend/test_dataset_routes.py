"""Dataset source inspection and fetch job admission routes."""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from dataset_route_fixtures import (
    CLEANUP,
    FETCH_JOBS,
    HUB_SOURCE,
    INSPECTIONS,
    PREFIX,
    SOURCE_KINDS,
    STORAGE,
    build_application,
    failure,
    fake_inspection,
    fetch_body,
    upload_archive,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_service import dataset_source_service
from backend_service.dataset_source_service import DatasetSourceService
from schemas.dataset_sources import DatasetInspection
from schemas.dataset_storage import DatasetStorageSummary


@pytest.fixture
def application(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Build an API on isolated folders whose scheduler never starts a job."""
    return build_application(tmp_path, monkeypatch)


@pytest.fixture
def client(application: FastAPI) -> Iterator[TestClient]:
    """Serve the application for one test."""
    with TestClient(application, raise_server_exceptions=False) as connection:
        yield connection


def test_inspection_reports_the_source_and_keeps_gated_sources_at_200(
    application: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route hands the request to the source layer and returns its answer."""
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake(spec: Any, **options: Any) -> DatasetInspection:
        """Answer the first call as available and later calls as gated."""
        calls.append((spec, options))
        return fake_inspection("available" if len(calls) == 1 else "access_required")

    monkeypatch.setattr(dataset_source_service, "inspect_source", fake)
    body = {"source": HUB_SOURCE, "source_name": "tiny", "maximum_images": 50}
    response = client.post(INSPECTIONS, json=body)
    assert response.status_code == 200
    assert DatasetInspection.model_validate_json(response.text) == fake_inspection()
    spec, options = calls[0]
    service: DatasetSourceService = application.state.dataset_sources
    assert spec.repository == "example/tiny"
    assert (options["source_name"], options["maximum_images"]) == ("tiny", 50)
    assert options["data_root"] == service.data_root == service.catalog.root / "data"
    assert options["cache_root"] == service.cache_root
    assert options["uploads"] is application.state.dataset_uploads
    gated = client.post(INSPECTIONS, json={"source": HUB_SOURCE})
    assert (gated.status_code, gated.json()["access"]) == (200, "access_required")
    capabilities = client.get(f"{PREFIX}/capabilities").json()
    assert capabilities["dataset_source_kinds"] == SOURCE_KINDS


def test_slow_inspection_times_out(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source that does not answer within the deadline is a 504."""

    def slow(spec: Any, **options: Any) -> DatasetInspection:
        """Answer only after the patched deadline has passed."""
        time.sleep(0.5)
        return fake_inspection()

    monkeypatch.setattr(dataset_source_service, "INSPECTION_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(dataset_source_service, "inspect_source", slow)
    response = client.post(INSPECTIONS, json={"source": HUB_SOURCE})
    assert failure(response) == (504, "inspection_timeout")


@pytest.mark.parametrize(
    "source",
    [
        {"source_kind": "https_archive", "url": "http://example.org/secret-plain.zip"},
        {
            "source_kind": "https_archive",
            "url": "https://u:secret-pw@example.org/a.zip",
        },
        {"source_kind": "hugging_face", "repository": "secret repository"},
    ],
)
def test_invalid_sources_are_refused_without_echo(
    client: TestClient, source: dict[str, str]
) -> None:
    """Plain http, credentials and malformed repositories fail validation."""
    body = {"source": {**source, "terms_reference": "https://example.org/terms"}}
    response = client.post(INSPECTIONS, json=body)
    assert failure(response) == (422, "invalid_request")
    assert "secret" not in response.text


def test_fetch_jobs_are_admitted_once_and_one_at_a_time(client: TestClient) -> None:
    """Replay, conflicting retries, incomplete uploads and a second job are handled."""
    incomplete = upload_archive(client, "upload-incomplete", complete=False)
    refused = client.post(FETCH_JOBS, json=fetch_body(incomplete))
    assert failure(refused) == (409, "upload_incomplete")
    upload = upload_archive(client, "upload-complete", complete=True)
    accepted = client.post(FETCH_JOBS, json=fetch_body(upload))
    assert accepted.status_code == 202
    job = accepted.json()
    assert (job["operation"], job["status"], job["phase"]) == (
        "fetch_dataset",
        "queued",
        "queued",
    )
    assert job["available_actions"] == ["cancel"]
    assert job["frozen_settings"]["source_name"] == "browser_zip"
    replay = client.post(FETCH_JOBS, json=fetch_body(upload))
    assert replay.status_code == 202
    assert replay.json()["job_identifier"] == job["job_identifier"]
    changed = client.post(FETCH_JOBS, json=fetch_body(upload, dataset_name="other"))
    assert failure(changed) == (409, "request_identifier_conflict")
    second = client.post(FETCH_JOBS, json=fetch_body(upload, "fetch-2"))
    assert failure(second) == (409, "dataset_fetch_busy")
    storage = DatasetStorageSummary.model_validate_json(client.get(STORAGE).text)
    assert (storage.active_fetch_jobs, storage.cleanup_available) == (1, False)
    assert failure(client.delete(CLEANUP)) == (409, "cache_busy")
    assert client.get(f"{PREFIX}/jobs/{job['job_identifier']}").status_code == 200
