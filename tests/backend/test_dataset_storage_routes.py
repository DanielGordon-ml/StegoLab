"""Dataset storage summary and unused download cleanup routes."""

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from dataset_route_fixtures import (
    ASSET_SHA256,
    CLEANUP,
    CREATED_AT,
    FETCH_JOBS,
    MARKER_REFERENCE,
    MARKER_REVISION,
    PREFIX,
    STORAGE,
    build_application,
    failure,
    fetch_body,
    marked_folder,
    planned_asset,
    upload_archive,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_service.dataset_source_service import DatasetSourceService
from backend_service.dataset_sources.cache import DatasetCache
from backend_service.dataset_sources.plans import asset_identity
from backend_service.dataset_sources.source_marker import MARKER_NAME
from schemas.dataset_cache import DatasetCacheSummary
from schemas.dataset_storage import DatasetStorageSummary

OTHER_OWNER = "other_owner"


@pytest.fixture
def application(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Build an API on isolated folders whose scheduler never starts a job."""
    return build_application(tmp_path, monkeypatch)


@pytest.fixture
def client(application: FastAPI) -> Iterator[TestClient]:
    """Serve the application for one test."""
    with TestClient(application, raise_server_exceptions=False) as connection:
        yield connection


def storage(client: TestClient) -> DatasetStorageSummary:
    """Read the storage summary through the route."""
    return DatasetStorageSummary.model_validate_json(client.get(STORAGE).text)


def marker_identity(expected_sha256: str | None) -> str:
    """Name the marked folder's archive the way a fetch would plan it."""
    return asset_identity(
        "https_archive", MARKER_REFERENCE, MARKER_REVISION, "fake.zip", expected_sha256
    )


def test_storage_summary_counts_marked_folders_and_cache_users(
    application: FastAPI, client: TestClient
) -> None:
    """Only folders with a marker count, and their assets are the cache users."""
    service: DatasetSourceService = application.state.dataset_sources
    marked_folder(service.data_root, "fake_source")
    (service.data_root / "unmarked").mkdir()
    (service.data_root / "unmarked" / "loose.bin").write_bytes(b"y" * 10)
    summary = storage(client)
    extra = (service.data_root / "fake_source" / MARKER_NAME).stat().st_size
    assert (summary.raw_source_folders, summary.raw_source_bytes) == (1, 1234 + extra)
    assert (summary.cache_entries, summary.cache_bytes) == (0, 0)
    assert (summary.prepared_revisions, summary.active_fetch_jobs) == (0, 0)
    assert summary.cleanup_available is False
    assert summary.free_disk_bytes > 0
    assert summary.minimum_free_bytes >= 10 * 1024**3
    # The marker records the measured digest, the cache may have been named by
    # the digest expected before the download, which is usually unknown.
    assert service.in_use() == {
        marker_identity(ASSET_SHA256): ["data/fake_source"],
        marker_identity(None): ["data/fake_source"],
    }


def test_cleanup_keeps_archives_cached_without_an_expected_digest(
    application: FastAPI, client: TestClient, tmp_path: Path
) -> None:
    """An archive fetched without a typed digest stays in use by its raw folder."""
    service: DatasetSourceService = application.state.dataset_sources
    marked_folder(service.data_root, "fake_source")
    source = tmp_path / "fake.zip"
    source.write_bytes(b"z" * 1234)
    DatasetCache(service.cache_root).import_file(
        "https_archive",
        marker_identity(None),
        planned_asset("fake.zip", 1234),
        source,
        owner="finished_job",
        reference=MARKER_REFERENCE,
        revision=MARKER_REVISION,
        terms_reference="https://example.org/terms",
        created_at=CREATED_AT,
    )
    before = storage(client)
    assert (before.cache_entries, before.cache_unused_entries) == (1, 0)
    assert before.cleanup_available is False
    response = client.delete(CLEANUP)
    assert response.status_code == 200
    after = DatasetStorageSummary.model_validate_json(response.text)
    assert (after.cache_entries, after.cache_bytes) == (1, 1234)


def test_cleanup_removes_unused_entries_and_leftover_partials(
    application: FastAPI, client: TestClient, tmp_path: Path
) -> None:
    """Unused entries and orphaned partials go on request; a job's on cancel."""
    service: DatasetSourceService = application.state.dataset_sources
    cache = DatasetCache(service.cache_root)
    source = tmp_path / "asset.bin"
    source.write_bytes(b"d" * 512)
    cache.import_file(
        "https_archive",
        "d" * 64,
        planned_asset("asset.bin", 512),
        source,
        owner="old_owner",
        reference="https://example.org/asset.bin",
        revision="etag-2",
        terms_reference="https://example.org/terms",
        created_at=CREATED_AT,
    )
    before = storage(client)
    assert (before.cache_entries, before.cache_unused_entries) == (1, 1)
    assert (before.cache_bytes, before.cleanup_available) == (512, True)
    response = client.delete(CLEANUP)
    assert response.status_code == 200
    after = DatasetStorageSummary.model_validate_json(response.text)
    assert (after.cache_entries, after.cache_unused_bytes) == (0, 0)
    assert after.cleanup_available is False
    run_label = "0123456789abcdef0123456789abcdef"
    for identity, owner in (("e" * 64, run_label), ("f" * 64, OTHER_OWNER)):
        asset = planned_asset("a.zip", 10)
        cache.open_partial("https_archive", identity, asset, owner=owner)
    service.discard_partials(f"job_{run_label}")
    assert cache.partial_directories(run_label) == []
    kept = cache.partial_directories(OTHER_OWNER)
    assert kept == [f"https_archive/{'f' * 64}.partial"]
    service.discard_partials(f"job_{run_label}")
    assert cache.partial_directories(OTHER_OWNER) == kept
    # A partial nobody owns any more is offered for cleanup and then removed.
    assert (storage(client).cleanup_available, storage(client).cache_entries) == (
        True,
        1,
    )
    assert client.delete(CLEANUP).status_code == 200
    assert cache.partial_directories(OTHER_OWNER) == []
    assert storage(client).cleanup_available is False


def test_cleanup_releases_the_queue_lock_while_deleting(
    application: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job actions keep answering during a slow cleanup, which refuses new fetches."""
    service: DatasetSourceService = application.state.dataset_sources
    upload = upload_archive(client, "upload-complete", complete=True)
    job = client.post(FETCH_JOBS, json=fetch_body(upload)).json()
    action = {"client_request_identifier": "cancel-1", "action": "cancel"}
    actions = f"{PREFIX}/jobs/{job['job_identifier']}/actions"
    assert client.post(actions, json=action).json()["status"] == "cancelled"
    started, release = threading.Event(), threading.Event()
    original = DatasetCache.clear_unused

    def slow_clear(
        self: DatasetCache, in_use: dict[str, list[str]], *, active_owners: set[str]
    ) -> DatasetCacheSummary:
        """Signal the test, wait until it is released, then clear for real."""
        started.set()
        assert release.wait(5)
        return original(self, in_use, active_owners=active_owners)

    monkeypatch.setattr(DatasetCache, "clear_unused", slow_clear)
    outcomes: list[DatasetStorageSummary | Exception] = []

    def cleanup() -> None:
        """Run the cleanup the way the route does and keep its outcome."""
        try:
            outcomes.append(service.remove_unused())
        except Exception as problem:
            outcomes.append(problem)

    thread = threading.Thread(target=cleanup)
    thread.start()
    assert started.wait(5)
    try:
        assert service.jobs.lock.acquire(timeout=1)
        service.jobs.lock.release()
        # The retried action replays under the queue lock and still answers.
        assert client.post(actions, json=action).status_code == 200
        refused = client.post(FETCH_JOBS, json=fetch_body(upload, "fetch-2"))
        assert failure(refused) == (409, "dataset_cleanup_busy")
        assert failure(client.delete(CLEANUP)) == (409, "dataset_cleanup_busy")
    finally:
        release.set()
        thread.join(5)
    assert isinstance(outcomes[0], DatasetStorageSummary)
    assert service.jobs.cache_cleanup_in_progress is False
    admitted = client.post(FETCH_JOBS, json=fetch_body(upload, "fetch-3"))
    assert admitted.status_code == 202
