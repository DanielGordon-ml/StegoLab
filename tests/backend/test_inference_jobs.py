"""Real encode and decode round trips that never persist a message or password."""

import json
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from inference_fixtures import (
    PREFIX,
    image_bytes,
    install_fixture_model,
    submit,
    upload,
    wait_for_job,
)
from PIL import Image

from backend_service.application import create_application
from backend_service.protocol_failures import RECOVERY_MESSAGE
from schemas.inference_jobs import (
    DecodingJobResult,
    EncodingJobResult,
)

MESSAGE = "Sunny garden — שלום \U0001f33b " + "sentinel-text"
PASSWORD = "sentinel-password-\U0001f511"


def assert_secrets_absent(root: Path) -> None:
    """Search every saved file for the message and password in any encoding."""
    probes = {MESSAGE, PASSWORD, json.dumps(MESSAGE)[1:-1], json.dumps(PASSWORD)[1:-1]}
    for path in root.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            for probe in probes:
                assert probe.encode("utf-8") not in content, path


def test_real_round_trip_keeps_secrets_in_memory_only(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Encode a Unicode message, download the PNG, decode it, and expire the text."""
    application = create_application(
        tmp_path / "data", tmp_path / "logs", fixture_workspace
    )
    with TestClient(application, raise_server_exceptions=False) as client:
        model = install_fixture_model(client)
        cover = upload(client, image_bytes((513, 517))).json()["image_reference"]
        body: dict[str, object] = {
            "client_request_identifier": "encode-1",
            "image_reference": cover,
            "model_identifier": model,
            "message": MESSAGE,
            "password": PASSWORD,
        }
        accepted = submit(client, "encoding", body)
        assert accepted.status_code == 202, accepted.text
        job = accepted.json()
        assert job["status"] == "queued" and job["operation"] == "encode"
        assert set(job["frozen_settings"]) == {
            "operation",
            "image_reference",
            "model_identifier",
            "message_byte_count",
            "width",
            "height",
        }
        assert (
            submit(client, "encoding", body).json()["job_identifier"]
            == (job["job_identifier"])
        )
        final = wait_for_job(client, job["job_identifier"])
        assert final.status == "completed", final.error
        assert final.result is not None
        result = EncodingJobResult.model_validate(final.result)
        assert (result.width, result.height) == (513, 517)
        assert result.message_byte_count == len(MESSAGE.encode("utf-8"))
        download = client.get(f"{PREFIX}/artifacts/{result.artifact_identifier}")
        assert download.status_code == 200
        assert download.headers["content-disposition"] == (
            f'attachment; filename="{result.filename}"'
        )
        assert len(download.content) == result.png_bytes
        with Image.open(BytesIO(download.content)) as image:
            assert image.size == (513, 517) and image.info == {}
        encoded = upload(client, download.content, purpose="encoded").json()
        decode_body: dict[str, object] = {
            "client_request_identifier": "decode-1",
            "image_reference": encoded["image_reference"],
            "model_identifier": model,
            "password": PASSWORD,
        }
        decoded = wait_for_job(
            client, submit(client, "decoding", decode_body).json()["job_identifier"]
        )
        assert decoded.status == "completed", decoded.error
        assert decoded.result is not None
        announced = DecodingJobResult.model_validate_json(json.dumps(decoded.result))
        text_path = f"{PREFIX}/jobs/{decoded.job_identifier}/decoded_text"
        text = client.get(text_path)
        assert text.status_code == 200
        assert text.headers["cache-control"] == "no-store"
        assert text.json()["text"] == MESSAGE
        assert text.json()["byte_count"] == announced.text_byte_count
        assert client.delete(text_path).status_code == 204
        expired = client.get(text_path)
        assert expired.status_code == 404
        assert expired.json()["error"]["code"] == "result_expired"
        assert client.delete(text_path).status_code == 204
        wrong = wait_for_job(
            client,
            submit(
                client,
                "decoding",
                {
                    **decode_body,
                    "client_request_identifier": "decode-2",
                    "password": "x",
                },
            ).json()["job_identifier"],
        )
        assert wrong.status == "failed"
        assert wrong.error is not None
        assert wrong.error["code"] == "message_recovery_failed"
        assert wrong.error["message"] == RECOVERY_MESSAGE
        unavailable = client.get(f"{PREFIX}/jobs/{wrong.job_identifier}/decoded_text")
        assert unavailable.status_code == 409
        listing = client.get(f"{PREFIX}/jobs").text
        assert MESSAGE not in listing and PASSWORD not in listing
    assert_secrets_absent(tmp_path)
