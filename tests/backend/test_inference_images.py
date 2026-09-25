"""Accept bounded image uploads, keep encoded PNG bytes exact, and expire them."""

import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from inference_fixtures import PREFIX, image_bytes, upload
from PIL import Image

from backend_service import inference_files
from backend_service.application import create_application
from schemas.capabilities import MAXIMUM_UPLOAD_BYTES
from schemas.errors import ErrorEnvelope


def test_cover_upload_prepares_png_and_serves_exact_bytes(
    client: TestClient, tmp_path: Path
) -> None:
    """Prepare covers into metadata-free PNG files and download them unchanged."""
    response = upload(client, image_bytes(orientation=6))
    assert response.status_code == 201, response.text
    body = response.json()
    reference = body["image_reference"]
    assert reference.startswith("image_") and len(reference) == 38
    assert body["purpose"] == "cover"
    assert body["summary"]["source_format"] == "PNG"
    assert body["summary"]["pixel_policy"] == "prepare_srgb"
    assert (body["summary"]["prepared_width"], body["summary"]["prepared_height"]) == (
        517,
        513,
    )
    assert body["warnings"] == ["The image was rotated to its displayed orientation."]
    created = datetime.fromisoformat(body["created_at"])
    assert datetime.fromisoformat(body["expires_at"]) - created == timedelta(hours=24)
    download = client.get(f"{PREFIX}/artifacts/{reference}")
    assert download.status_code == 200
    assert download.headers["content-type"] == "image/png"
    assert download.headers["cache-control"] == "no-store"
    assert download.headers["x-content-type-options"] == "nosniff"
    assert download.headers["content-disposition"] == (
        f'attachment; filename="stegolab-cover-{reference[6:14]}.png"'
    )
    with Image.open(BytesIO(download.content)) as stored:
        assert stored.size == (517, 513)
        assert stored.info == {}
    directory = tmp_path / "data" / "inference" / "images" / reference
    assert (directory / "image.png").read_bytes() == download.content
    record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
    assert set(record) == set(body) | {"stored_bytes", "checksum"}
    jpeg = upload(client, image_bytes(image_format="JPEG"), content_type="image/jpeg")
    assert jpeg.status_code == 201, jpeg.text
    assert jpeg.json()["summary"]["source_format"] == "JPEG"
    assert jpeg.json()["warnings"] == [
        "The JPEG file was converted to PNG. Encode and Decode always use PNG."
    ]


def test_encoded_upload_keeps_exact_bytes_and_rejects_jpeg(client: TestClient) -> None:
    """Store decode inputs verbatim so stored pixels are what the decoder reads."""
    data = image_bytes()
    response = upload(client, data, purpose="encoded")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["purpose"] == "encoded"
    assert body["summary"]["pixel_policy"] == "preserve_stored"
    assert body["warnings"] == []
    download = client.get(f"{PREFIX}/artifacts/{body['image_reference']}")
    assert download.content == data
    assert download.headers["content-disposition"].startswith(
        'attachment; filename="stegolab-encoded-'
    )
    jpeg = upload(
        client,
        image_bytes(image_format="JPEG"),
        purpose="encoded",
        content_type="image/jpeg",
    )
    assert jpeg.status_code == 422
    assert jpeg.json()["error"]["code"] == "image_purpose_format"


@pytest.mark.parametrize(
    ("data", "purpose", "content_type", "status", "code"),
    [
        (
            b"\x89PNG" + b"\0" * MAXIMUM_UPLOAD_BYTES,
            "cover",
            "image/png",
            413,
            "upload_too_large",
        ),
        (image_bytes(), "cover", "application/json", 415, "upload_content_type"),
        (image_bytes(), "secret", "image/png", 422, "invalid_request"),
        (image_bytes(mode="L"), "cover", "image/png", 422, "image_invalid"),
        (image_bytes((300, 300)), "cover", "image/png", 422, "image_limits"),
        (b"not an image at all", "cover", "image/png", 422, "image_invalid"),
        (b"", "cover", "image/png", 422, "image_invalid"),
    ],
)
def test_uploads_are_bounded_and_validated(
    client: TestClient,
    tmp_path: Path,
    data: bytes,
    purpose: str,
    content_type: str,
    status: int,
    code: str,
) -> None:
    """Refuse oversized, mislabeled, and unsupported uploads without storing them."""
    response = upload(client, data, purpose=purpose, content_type=content_type)
    assert response.status_code == status, response.text
    envelope = ErrorEnvelope.model_validate_json(response.text)
    assert envelope.error.code == code
    assert list((tmp_path / "data" / "inference" / "images").iterdir()) == []


def test_storage_cap_expiry_sweep_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse uploads past the cap, forget expired images, and detect changed files."""
    application = create_application(tmp_path / "data", tmp_path / "logs")
    images = tmp_path / "data" / "inference" / "images"
    with TestClient(application, raise_server_exceptions=False) as client:
        monkeypatch.setattr(inference_files, "STORAGE_CAP_BYTES", 1)
        full = upload(client, image_bytes())
        assert full.status_code == 503
        assert full.json()["error"]["code"] == "inference_storage_full"
        assert list(images.iterdir()) == []
        monkeypatch.setattr(inference_files, "STORAGE_CAP_BYTES", 512 * 1024**2)
        old = upload(client, image_bytes()).json()["image_reference"]
        store = application.state.inference_files
        store.clock = lambda: datetime.now(UTC) + timedelta(hours=25)
        expired = client.get(f"{PREFIX}/artifacts/{old}")
        assert expired.status_code == 404
        assert expired.json()["error"]["code"] == "artifact_expired"
        capacity = client.post(
            f"{PREFIX}/capacity",
            json={"image_reference": old, "model_identifier": "any"},
        )
        assert capacity.status_code == 404
        assert capacity.json()["error"]["code"] == "image_expired"
        store.clock = lambda: datetime.now(UTC)
        assert client.get(f"{PREFIX}/artifacts/{old}").status_code == 200
        store.clock = lambda: datetime.now(UTC) + timedelta(hours=25)
        recent = upload(client, image_bytes()).json()["image_reference"]
        assert {path.name for path in images.iterdir()} == {recent}
        for bad in ("image_zz", "image_" + "0" * 32, "..", "image_%2e%2e"):
            assert client.get(f"{PREFIX}/artifacts/{bad}").status_code == 404
        with (images / recent / "image.png").open("ab") as stream:
            stream.write(b"\0")
        tampered = client.get(f"{PREFIX}/artifacts/{recent}")
        assert tampered.status_code == 404
        assert tampered.json()["error"]["code"] == "artifact_expired"
    with TestClient(create_application(tmp_path / "data", tmp_path / "logs")) as client:
        assert {path.name for path in images.iterdir()} == {recent}


def test_chunked_upload_is_bounded_while_streaming(
    client: TestClient, tmp_path: Path
) -> None:
    """Stop reading a body without a declared length once it passes the limit."""
    chunks = [b"\x89PNG" + b"\0" * 65_532] * (MAXIMUM_UPLOAD_BYTES // 65_536 + 1)
    response = client.post(
        f"{PREFIX}/images?purpose=cover",
        content=iter(chunks),
        headers={"Content-Type": "image/png", "Transfer-Encoding": "chunked"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"
    assert list((tmp_path / "data" / "inference" / "images").iterdir()) == []
