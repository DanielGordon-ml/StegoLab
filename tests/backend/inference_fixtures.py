"""Shared helpers for upload, capacity, and inference tests over the HTTP API."""

import time
from io import BytesIO
from typing import Literal, cast

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from backend_service.model_fixture_channel import FIXTURE_PACKAGE_NAME
from schemas.jobs import JobSnapshot

PREFIX = "/api/v1"


def image_bytes(
    size: tuple[int, int] = (513, 517),
    *,
    image_format: Literal["PNG", "JPEG"] = "PNG",
    mode: Literal["RGB", "RGBA", "L"] = "RGB",
    orientation: int = 1,
) -> bytes:
    """Encode a small test image with an optional orientation tag."""
    color: int | tuple[int, ...] = 90
    if mode != "L":
        color = (17, 61, 209) if mode == "RGB" else (17, 61, 209, 255)
    stream = BytesIO()
    with Image.new(mode, size, color) as image:
        image.putpixel((min(400, size[0] - 1), min(300, size[1] - 1)), color)
        metadata = Image.Exif()
        metadata[274] = orientation
        image.save(stream, format=image_format, exif=metadata, quality=95)
    return stream.getvalue()


def upload(
    client: TestClient,
    data: bytes,
    purpose: str = "cover",
    content_type: str = "image/png",
) -> httpx.Response:
    """Send one raw image body with the given purpose and content type."""
    return cast(
        httpx.Response,
        client.post(
            f"{PREFIX}/images",
            params={"purpose": purpose},
            content=data,
            headers={"Content-Type": content_type},
        ),
    )


def install_fixture_model(client: TestClient) -> str:
    """Install the labelled fixture pair from the workspace and return its name."""
    workspace = client.get(f"{PREFIX}/workspace").json()
    reference = next(
        item["identifier"]
        for item in workspace["exports"]
        if item["name"] == FIXTURE_PACKAGE_NAME and item["kind"] == "encoder"
    )
    response = client.post(
        f"{PREFIX}/models/install",
        json={"client_request_identifier": "install", "export_reference": reference},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["model_identifier"])


def wait_for_job(
    client: TestClient, identifier: str, seconds: float = 120
) -> JobSnapshot:
    """Wait a bounded time for a real package process to publish its final record."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        response = client.get(f"{PREFIX}/jobs/{identifier}")
        assert response.status_code == 200, response.text
        snapshot = JobSnapshot.model_validate_json(response.text)
        if snapshot.status not in ("queued", "running"):
            return snapshot
        time.sleep(0.1)
    raise AssertionError("The inference job did not finish in time.")


def submit(
    client: TestClient, operation: str, body: dict[str, object]
) -> httpx.Response:
    """Send one encode or decode job request."""
    return cast(httpx.Response, client.post(f"{PREFIX}/{operation}_jobs", json=body))
