"""Exercise transaction safety, restart behavior, and corruption handling."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend_service.application import create_application
from backend_service.failures import StorageFailure
from backend_service.storage import StateStore
from schemas.configuration import ConfigurationProfile
from schemas.jobs import JobSnapshot


def test_concurrent_identical_requests_commit_once(tmp_path: Path) -> None:
    """Serialize concurrent retry writes and store one stable result."""
    store = StateStore(tmp_path)
    configuration = ConfigurationProfile(checkpoint_interval_seconds=600)

    def save(_: int) -> ConfigurationProfile:
        """Submit the same request from a concurrent worker."""
        return store.update_configuration("same-request", configuration)

    with ThreadPoolExecutor(max_workers=8) as workers:
        assert list(workers.map(save, range(16))) == [configuration] * 16
    assert StateStore(tmp_path).get_configuration() == configuration
    with sqlite3.connect(store.database_path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM mutation_results").fetchone()[0]
            == 1
        )
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"schema_version":1}',
        '{"checkpoint_interval_seconds":300}',
        "not-valid-json-secret-sentinel",
        "null",
        '{"schema_version":1,"checkpoint_interval_seconds":"secret-sentinel"}',
    ],
)
def test_corrupt_saved_configuration_is_never_replaced(
    tmp_path: Path, payload: str
) -> None:
    """Fail startup and mutations without silently saving default values."""
    store = StateStore(tmp_path / "data")
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("UPDATE configuration SET payload=?", (payload,))
    with pytest.raises(StorageFailure):
        store.update_configuration("cannot-overwrite", ConfigurationProfile())
    with pytest.raises(RuntimeError, match="saved data is unavailable or invalid"):
        with TestClient(create_application(tmp_path / "data", tmp_path / "logs")):
            pytest.fail("Corrupt saved state must prevent startup.")
    with sqlite3.connect(store.database_path) as connection:
        assert (
            connection.execute("SELECT payload FROM configuration").fetchone()[0]
            == payload
        )
    logs = "".join(path.read_text() for path in (tmp_path / "logs").rglob("*.jsonl"))
    assert "secret-sentinel" not in logs


def test_unsupported_storage_version_is_preserved(tmp_path: Path) -> None:
    """Require explicit migration support for an unknown database version."""
    store = StateStore(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("PRAGMA user_version=2")
    with pytest.raises(StorageFailure):
        StateStore(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_jobs_survive_restart_and_remain_metadata_only(tmp_path: Path) -> None:
    """Load stored snapshots through the API after the store is recreated."""
    now = datetime.now(UTC)
    snapshot = JobSnapshot(
        job_identifier="fixture-job",
        status="interrupted",
        phase="saving",
        configuration=ConfigurationProfile(),
        available_actions=["resume"],
        created_at=now,
        updated_at=now,
        progress=0.5,
    )
    store = StateStore(tmp_path / "data")
    store.save_job(snapshot)
    assert StateStore(tmp_path / "data").get_job("fixture-job") == snapshot
    with TestClient(create_application(tmp_path / "data", tmp_path / "logs")) as client:
        assert client.get("/api/v1/jobs").json() == {
            "items": [snapshot.model_dump(mode="json")]
        }
        response = client.get("/api/v1/jobs/fixture-job")
        assert response.json() == snapshot.model_dump(mode="json")
        assert "password" not in response.text and "plaintext" not in response.text
        with sqlite3.connect(store.database_path) as connection:
            connection.execute("UPDATE jobs SET payload='secret-corrupt-job'")
        response = client.get("/api/v1/jobs")
        assert response.status_code == 503
        assert "secret-corrupt-job" not in response.text


def test_missing_configuration_is_not_recreated(tmp_path: Path) -> None:
    """Preserve a damaged database instead of inserting new defaults."""
    store = StateStore(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DELETE FROM configuration")
    with pytest.raises(StorageFailure):
        StateStore(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM configuration").fetchone()[0] == 0
        )


def test_runtime_missing_storage_returns_safe_failure(tmp_path: Path) -> None:
    """Translate a missing storage directory into a recoverable response."""
    application = create_application(tmp_path / "data", tmp_path / "logs")
    with TestClient(application) as client:
        (tmp_path / "data").rename(tmp_path / "data-kept")
        response = client.get("/api/v1/configuration")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "storage_unavailable"
        (tmp_path / "data-kept").rename(tmp_path / "data")
        assert (
            client.get("/api/v1/configuration").json()
            == ConfigurationProfile().model_dump()
        )
