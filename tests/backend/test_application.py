"""Check API contracts, persistence across lifespans, and safe failures."""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend_service.application import create_application
from schemas.configuration import ConfigurationProfile
from schemas.errors import ErrorEnvelope

PREFIX = "/api/v1"


def update_body(identifier: str, interval: int = 600) -> dict[str, object]:
    """Build a complete settings mutation for API tests."""
    return {
        "client_request_identifier": identifier,
        "configuration": {"schema_version": 1, "checkpoint_interval_seconds": interval},
    }


def test_initial_capabilities_and_empty_lists(client: TestClient) -> None:
    """Expose ready CPU operation while model features and job lists stay empty."""
    assert client.get(f"{PREFIX}/health").json() == {
        "status": "ready",
        "application_version": "0.1.0",
    }
    assert client.get(f"{PREFIX}/capabilities").json() == {
        "application_version": "0.1.0",
        "available_devices": ["cpu"],
        "available_models": [],
        "available_profiles": [],
        "encoding_available": False,
        "decoding_available": False,
        "training_available": True,
        "maximum_payload_bytes": 0,
    }
    for collection in ("models", "jobs"):
        assert client.get(f"{PREFIX}/{collection}").json() == {"items": []}
    assert client.get(f"{PREFIX}/jobs/missing").status_code == 404


@pytest.mark.parametrize("value", [0, -60, 61, 60.0, True, "300", None])
def test_checkpoint_interval_is_strict(client: TestClient, value: object) -> None:
    """Reject conversions, fractions of minutes, and nonpositive intervals."""
    response = client.put(
        f"{PREFIX}/configuration",
        json={
            "client_request_identifier": "invalid-interval",
            "configuration": {
                "schema_version": 1,
                "checkpoint_interval_seconds": value,
            },
        },
    )
    assert response.status_code == 422
    ErrorEnvelope.model_validate_json(response.text)
    assert (
        client.get(f"{PREFIX}/configuration").json()["checkpoint_interval_seconds"]
        == 300
    )


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_version_is_strict(client: TestClient, version: object) -> None:
    """Accept only the integer version supported by this release."""
    response = client.put(
        f"{PREFIX}/configuration",
        json={
            "client_request_identifier": "invalid-version",
            "configuration": {
                "schema_version": version,
                "checkpoint_interval_seconds": 300,
            },
        },
    )
    assert response.status_code == 422


def test_save_restart_retry_and_reset(tmp_path: Path) -> None:
    """Keep settings and original mutation results across backend restarts."""
    for attempt in range(2):
        application = create_application(tmp_path / "data", tmp_path / "logs")
        with TestClient(application) as client:
            if attempt == 0:
                assert (
                    client.put(
                        f"{PREFIX}/configuration", json=update_body("first-save")
                    ).json()["checkpoint_interval_seconds"]
                    == 600
                )
                continue
            assert (
                client.get(f"{PREFIX}/configuration").json()[
                    "checkpoint_interval_seconds"
                ]
                == 600
            )
            client.put(f"{PREFIX}/configuration", json=update_body("second-save", 900))
            original = client.put(
                f"{PREFIX}/configuration", json=update_body("first-save")
            )
            assert original.json()["checkpoint_interval_seconds"] == 600
            assert (
                client.get(f"{PREFIX}/configuration").json()[
                    "checkpoint_interval_seconds"
                ]
                == 900
            )
            reset = client.post(
                f"{PREFIX}/configuration/reset",
                json={"client_request_identifier": "reset-once"},
            )
            assert reset.json() == ConfigurationProfile().model_dump()
    with TestClient(create_application(tmp_path / "data", tmp_path / "logs")) as client:
        assert (
            client.get(f"{PREFIX}/configuration").json()
            == ConfigurationProfile().model_dump()
        )


def test_changed_request_and_operation_conflict(client: TestClient) -> None:
    """Prevent one request identifier from being used for unrelated changes."""
    client.put(f"{PREFIX}/configuration", json=update_body("reused"))
    different_body = client.put(
        f"{PREFIX}/configuration", json=update_body("reused", 900)
    )
    different_operation = client.post(
        f"{PREFIX}/configuration/reset", json={"client_request_identifier": "reused"}
    )
    assert different_body.status_code == different_operation.status_code == 409
    assert (
        client.get(f"{PREFIX}/configuration").json()["checkpoint_interval_seconds"]
        == 600
    )


def test_secrets_never_appear_in_errors_or_logs(
    client: TestClient, tmp_path: Path
) -> None:
    """Keep unknown fields, malformed JSON, identifiers, and input out of logs."""
    secret = "secret-sentinel-password-and-private-text"
    responses = [
        client.put(f"{PREFIX}/configuration", json={"password": secret}),
        client.put(f"{PREFIX}/configuration", content='{"secret":"' + secret),
        client.post(
            f"{PREFIX}/configuration/reset",
            json={"client_request_identifier": "secret-check", "plaintext": secret},
        ),
        client.get(f"{PREFIX}/jobs/{secret}"),
    ]
    for response in responses:
        assert response.status_code >= 400
        assert secret not in response.text
        ErrorEnvelope.model_validate_json(response.text)
    logs = "\n".join(path.read_text() for path in (tmp_path / "logs").rglob("*.jsonl"))
    assert secret not in logs
    for line in logs.splitlines():
        assert set(json.loads(line)) == {"time", "event", "diagnostic_reference"}


def test_failed_write_rolls_back(client: TestClient, tmp_path: Path) -> None:
    """Rollback a changed setting when saving its request result fails."""
    with sqlite3.connect(tmp_path / "data" / "stegolab.sqlite3") as connection:
        connection.execute(
            "CREATE TRIGGER fail_result BEFORE INSERT ON mutation_results "
            "BEGIN SELECT RAISE(ABORT, 'secret-database-error'); END"
        )
    response = client.put(f"{PREFIX}/configuration", json=update_body("failed-write"))
    assert response.status_code == 503
    assert "secret-database-error" not in response.text
    assert (
        client.get(f"{PREFIX}/configuration").json()
        == ConfigurationProfile().model_dump()
    )
    with sqlite3.connect(tmp_path / "data" / "stegolab.sqlite3") as connection:
        assert (
            connection.execute("SELECT count(*) FROM mutation_results").fetchone()[0]
            == 0
        )


def test_invalid_response_is_sanitized(tmp_path: Path) -> None:
    """Validate outgoing responses and keep invalid returned values private."""
    application = create_application(tmp_path / "data", tmp_path / "logs")

    @application.get("/invalid-response", response_model=ConfigurationProfile)
    def invalid_response() -> dict[str, object]:
        """Return a deliberately invalid record to exercise response validation."""
        return {"checkpoint_interval_seconds": "secret-invalid-response"}

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/invalid-response")
    assert response.status_code == 500
    assert "secret-invalid-response" not in response.text
    ErrorEnvelope.model_validate_json(response.text)
