"""Route contracts for encode and decode jobs with a stand-in package runner."""

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from inference_fixtures import (
    PREFIX,
    image_bytes,
    install_fixture_model,
    submit,
    upload,
    wait_for_job,
)

from backend_service import workflow_supervision
from backend_service.application import create_application
from backend_service.protocol_failures import recovery_failure
from schemas.errors import ErrorEnvelope


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Replace package processes with a copy-and-remember stand-in."""
    remembered: dict[str, str] = {}

    def fake_run_package(
        package: Path, arguments: list[str], *, secret_input: bytes, **_: Any
    ) -> bytes:
        """Copy the cover as the encoded PNG and echo the remembered message."""
        assert not any("sentinel" in argument for argument in arguments), arguments
        assert not any("sentinel" in value for value in os.environ.values())
        secrets = json.loads(secret_input)
        if arguments[0] == "encode":
            shutil.copyfile(arguments[2], arguments[4])
            remembered.update(secrets)
            return b""
        if secrets["password"] != remembered.get("password"):
            raise recovery_failure()
        return remembered["message"].encode("utf-8")

    monkeypatch.setattr(
        "backend_service.inference_runner.run_package", fake_run_package
    )
    return remembered


@pytest.fixture
def client(fixture_workspace: Path, tmp_path: Path) -> Callable[[], TestClient]:
    """Open an application over the shared fixture workspace on demand."""

    def open_client() -> TestClient:
        application = create_application(
            tmp_path / "data", tmp_path / "logs", fixture_workspace
        )
        return TestClient(application, raise_server_exceptions=False)

    return open_client


def encode_body(image: str, model: str, **changes: object) -> dict[str, object]:
    """Build a complete encode request with optional overrides."""
    body: dict[str, object] = {
        "client_request_identifier": "encode",
        "image_reference": image,
        "model_identifier": model,
        "message": "hello",
        "password": "sentinel-password",
    }
    return {**body, **changes}


def test_routes_validate_admit_and_replay_without_leaking_secrets(
    client: Callable[[], TestClient], fake_runner: dict[str, str], tmp_path: Path
) -> None:
    """Explain refusals in plain words and never echo a password or message."""
    with client() as api:
        model = install_fixture_model(api)
        cover = upload(api, image_bytes((512, 512))).json()["image_reference"]
        encoded = upload(api, image_bytes(), purpose="encoded").json()[
            "image_reference"
        ]
        cases = [
            (encode_body(cover, model, password="s" * 1025), 422, "invalid_request"),
            (encode_body(cover, model, message="m" * 1025), 422, "invalid_request"),
            (encode_body(cover, model, message="m" * 257), 422, "message_too_long"),
            (encode_body(encoded, model), 422, "image_purpose_mismatch"),
            (encode_body(cover, "missing"), 404, "model_not_installed"),
            (encode_body("image_" + "0" * 32, model), 404, "image_expired"),
            (encode_body(cover, model, secret="x"), 422, "invalid_request"),
        ]
        for body, status, code in cases:
            response = submit(api, "encoding", body)
            assert response.status_code == status, response.text
            assert ErrorEnvelope.model_validate_json(response.text).error.code == code
            assert "sentinel" not in response.text
        accepted = submit(api, "encoding", encode_body(cover, model))
        assert accepted.status_code == 202
        assert "sentinel" not in accepted.text and "hello" not in accepted.text
        replay = submit(api, "encoding", encode_body(cover, model))
        assert replay.json()["job_identifier"] == accepted.json()["job_identifier"]
        for changed in (
            encode_body(encoded, model),
            encode_body(cover, model, message="other"),
            encode_body(cover, model, password="other-password"),
        ):
            conflict = submit(api, "encoding", changed)
            assert conflict.status_code == 409, conflict.text
            assert conflict.json()["error"]["code"] == "request_identifier_conflict"
        final = wait_for_job(api, accepted.json()["job_identifier"], seconds=30)
        assert final.status == "completed", final.error
        assert final.result is not None
        artifact = api.get(f"{PREFIX}/artifacts/{final.result['artifact_identifier']}")
        assert artifact.status_code == 200
        text = api.get(f"{PREFIX}/jobs/{final.job_identifier}/decoded_text")
        assert text.status_code == 409
        assert api.get(f"{PREFIX}/jobs/job_missing/decoded_text").status_code == 404
        decoding = submit(
            api,
            "decoding",
            {
                "client_request_identifier": "decode",
                "image_reference": encoded,
                "model_identifier": model,
                "password": "sentinel-password",
            },
        )
        decoded = wait_for_job(api, decoding.json()["job_identifier"], seconds=30)
        assert decoded.status == "completed", decoded.error
        recovered = api.get(f"{PREFIX}/jobs/{decoded.job_identifier}/decoded_text")
        assert recovered.json()["text"] == "hello"
        assert fake_runner["password"] == "sentinel-password"
    for path in (tmp_path / "data").rglob("*"):
        if path.is_file():
            assert b"sentinel-password" not in path.read_bytes(), path


def test_queue_is_bounded_and_waiting_jobs_report_lost_inputs_after_restart(
    client: Callable[[], TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold eight jobs at most and explain lost secrets after an application restart."""
    monkeypatch.setattr(workflow_supervision, "next_job", lambda service: None)
    with client() as api:
        model = install_fixture_model(api)
        cover = upload(api, image_bytes()).json()["image_reference"]
        for index in range(8):
            response = submit(
                api,
                "encoding",
                encode_body(cover, model, client_request_identifier=f"job-{index}"),
            )
            assert response.status_code == 202, response.text
        ninth = submit(
            api,
            "encoding",
            encode_body(cover, model, client_request_identifier="ninth"),
        )
        assert ninth.status_code == 409
        assert ninth.json()["error"]["code"] == "inference_queue_full"
        queued = api.get(f"{PREFIX}/jobs").json()["items"]
        text = api.get(f"{PREFIX}/jobs/{queued[0]['job_identifier']}/decoded_text")
        assert text.status_code == 409
    with client() as api:
        for job in api.get(f"{PREFIX}/jobs").json()["items"]:
            assert job["status"] == "interrupted"
            assert job["error"]["code"] == "inputs_lost"
            assert "password" not in job["frozen_settings"]
