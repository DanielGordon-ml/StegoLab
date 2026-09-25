"""Report per-image capacity for an installed model and refuse other images."""

from pathlib import Path
from typing import cast

import httpx
from fastapi.testclient import TestClient
from inference_fixtures import PREFIX, image_bytes, install_fixture_model, upload

from backend_service.application import create_application
from backend_service.payload_capacity import calculate_capacity
from schemas.protocol import ProtocolContext


def capacity(client: TestClient, image: str, model: str) -> httpx.Response:
    """Ask for the message limit of one uploaded image with one model."""
    return cast(
        httpx.Response,
        client.post(
            f"{PREFIX}/capacity",
            json={"image_reference": image, "model_identifier": model},
        ),
    )


def test_capacity_matches_the_protocol_layout_inside_the_model_range(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Return the exact layout accounting for sizes the installed model accepts."""
    application = create_application(
        tmp_path / "data", tmp_path / "logs", fixture_workspace
    )
    with TestClient(application, raise_server_exceptions=False) as client:
        model = install_fixture_model(client)
        compatibility = client.get(f"{PREFIX}/models").json()["items"][0][
            "compatibility_identifier"
        ]
        for size in ((512, 512), (513, 517), (1024, 1024)):
            image = upload(client, image_bytes(size)).json()["image_reference"]
            response = capacity(client, image, model)
            assert response.status_code == 200, response.text
            expected = calculate_capacity(
                ProtocolContext(
                    width=size[0],
                    height=size[1],
                    compatibility_identifier=compatibility,
                )
            )
            assert response.json() == {
                "image_reference": image,
                "model_identifier": model,
                "profile_identifier": "test_only_v1",
                "width": size[0],
                "height": size[1],
                "maximum_message_bytes": expected.maximum_message_bytes,
                "capacity": expected.model_dump(),
            }
        assert response.json()["maximum_message_bytes"] == 1024


def test_capacity_refuses_out_of_range_images_and_unknown_references(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Explain the model range in plain words and hide missing items safely."""
    application = create_application(
        tmp_path / "data", tmp_path / "logs", fixture_workspace
    )
    with TestClient(application, raise_server_exceptions=False) as client:
        model = install_fixture_model(client)
        wide = upload(client, image_bytes((1100, 600), image_format="JPEG")).json()
        assert wide["summary"]["prepared_width"] == 1100
        outside = capacity(client, wide["image_reference"], model)
        assert outside.status_code == 422
        assert outside.json()["error"]["code"] == "image_outside_model_range"
        assert "between 512 and 1024 pixels" in outside.json()["error"]["message"]
        image = upload(client, image_bytes()).json()["image_reference"]
        missing_image = capacity(client, "image_" + "a" * 32, model)
        assert missing_image.status_code == 404
        assert missing_image.json()["error"]["code"] == "image_expired"
        missing_model = capacity(client, image, "not_installed")
        assert missing_model.status_code == 404
        assert missing_model.json()["error"]["code"] == "model_not_installed"
        malformed = capacity(client, "../secret", model)
        assert malformed.status_code == 422
        assert client.delete(f"{PREFIX}/models/{model}").status_code == 204
        assert capacity(client, image, model).status_code == 404
