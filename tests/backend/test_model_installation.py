"""Install verified experimental pairs explicitly and derive features from them."""

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_service.application import create_application
from backend_service.model_fixture_channel import (
    FIXTURE_PACKAGE_NAME,
    FIXTURE_SOURCE_IDENTIFIER,
)
from schemas.configuration import ConfigurationProfile
from schemas.jobs import JobSnapshot
from schemas.workflows import WorkflowRequest

PREFIX = "/api/v1"


def application_for(tmp_path: Path, workspace: Path) -> FastAPI:
    """Build an API over private state directories and the given workspace."""
    return create_application(tmp_path / "data", tmp_path / "logs", workspace)


def start(tmp_path: Path, workspace: Path) -> TestClient:
    """Open a test client over a freshly built application."""
    return TestClient(
        application_for(tmp_path, workspace), raise_server_exceptions=False
    )


def export_reference(client: TestClient, kind: str = "encoder") -> str:
    """Find the registered reference for one role of the fixture pair."""
    workspace = client.get(f"{PREFIX}/workspace").json()
    for item in workspace["exports"]:
        if item["name"] == FIXTURE_PACKAGE_NAME and item["kind"] == kind:
            return str(item["identifier"])
    raise AssertionError("The fixture export was not discovered.")


def install(client: TestClient, identifier: str, reference: str) -> httpx.Response:
    """Send one installation request with a retry identifier."""
    return cast(
        httpx.Response,
        client.post(
            f"{PREFIX}/models/install",
            json={
                "client_request_identifier": identifier,
                "export_reference": reference,
            },
        ),
    )


def copied_workspace(source: Path, destination: Path) -> Path:
    """Copy the fixture package into a fresh workspace that tests may damage."""
    shutil.copytree(
        source / "models" / FIXTURE_PACKAGE_NAME,
        destination / "models" / FIXTURE_PACKAGE_NAME,
    )
    return destination


def test_install_flips_capabilities_replays_and_persists(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Advertise inference only after an explicit, verified installation."""
    with start(tmp_path, fixture_workspace) as client:
        before = client.get(f"{PREFIX}/capabilities").json()
        assert before["encoding_available"] is False
        assert before["available_models"] == []
        reference = export_reference(client)
        first = install(client, "install-1", reference)
        assert first.status_code == 201
        model = first.json()
        assert model["model_identifier"] == FIXTURE_PACKAGE_NAME
        assert model["source_identifier"] == FIXTURE_SOURCE_IDENTIFIER
        assert model["status"] == "experimental"
        assert model["maximum_side"] == 1024
        replay = install(client, "install-1", reference)
        assert replay.status_code == 200
        assert replay.json() == model
        conflict = install(client, "install-1", "another-reference")
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "request_identifier_conflict"
        same_pair = install(client, "install-2", export_reference(client, "decoder"))
        assert same_pair.status_code == 200
        assert (
            same_pair.json()["compatibility_identifier"]
            == (model["compatibility_identifier"])
        )
        after = client.get(f"{PREFIX}/capabilities").json()
        assert after["encoding_available"] is True
        assert after["decoding_available"] is True
        assert after["available_models"] == [FIXTURE_PACKAGE_NAME]
        assert after["available_profiles"] == ["test_only_v1"]
        assert after["maximum_payload_bytes"] == 1024
        assert (after["minimum_image_side"], after["maximum_image_side"]) == (512, 1024)
        assert after["experimental_models_only"] is True
        assert client.get(f"{PREFIX}/models").json() == {"items": [model]}
    with start(tmp_path, fixture_workspace) as client:
        assert client.get(f"{PREFIX}/models").json() == {"items": [model]}
        assert client.get(f"{PREFIX}/capabilities").json()["encoding_available"]


def test_install_refuses_unverified_incomplete_or_unknown_pairs(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Never install a pair without both packages and a matching verification."""
    workspace = copied_workspace(fixture_workspace, tmp_path / "workspace")
    package = workspace / "models" / FIXTURE_PACKAGE_NAME
    with start(tmp_path, workspace) as client:
        reference = export_reference(client)
        unknown = install(client, "unknown", "missing-reference")
        assert unknown.status_code == 404
        (package / "verification.json").unlink()
        unverified = install(client, "unverified", reference)
        assert unverified.status_code == 422
        assert unverified.json()["error"]["code"] == "model_verification_missing"
        shutil.rmtree(package / "decoder")
        incomplete = install(client, "incomplete", reference)
        assert incomplete.status_code == 422
        assert incomplete.json()["error"]["code"] == "model_pair_incomplete"
        assert client.get(f"{PREFIX}/models").json() == {"items": []}
        assert (
            client.get(f"{PREFIX}/capabilities").json()["encoding_available"] is False
        )


def test_remove_refuses_while_in_use_then_forgets_the_model(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Keep a model installed while a queued inference job still references it."""
    application = application_for(tmp_path, fixture_workspace)
    with TestClient(application, raise_server_exceptions=False) as client:
        assert install(client, "install", export_reference(client)).status_code == 201
        store = application.state.workspace_jobs.store
        now = datetime.now(UTC)
        job = JobSnapshot(
            job_identifier="job_encode",
            status="queued",
            phase="queued",
            configuration=ConfigurationProfile(),
            available_actions=["cancel"],
            created_at=now,
            updated_at=now,
            operation="encode",
            frozen_settings={"model_identifier": FIXTURE_PACKAGE_NAME},
        )
        store.save(
            job,
            request=WorkflowRequest(client_request_identifier="job", operation="train"),
        )
        busy = client.delete(f"{PREFIX}/models/{FIXTURE_PACKAGE_NAME}")
        assert busy.status_code == 409
        assert busy.json()["error"]["code"] == "model_in_use"
        store.save(job.model_copy(update={"status": "cancelled", "phase": "cancelled"}))
        assert (
            client.delete(f"{PREFIX}/models/{FIXTURE_PACKAGE_NAME}").status_code == 204
        )
        assert client.get(f"{PREFIX}/models").json() == {"items": []}
        assert (
            client.get(f"{PREFIX}/capabilities").json()["encoding_available"] is False
        )
        assert (
            client.delete(f"{PREFIX}/models/{FIXTURE_PACKAGE_NAME}").status_code == 404
        )
        assert client.delete(f"{PREFIX}/models/..").status_code in (404, 422)


def test_vanished_package_is_not_advertised(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Stop advertising a model whose package directory disappeared."""
    workspace = copied_workspace(fixture_workspace, tmp_path / "workspace")
    with start(tmp_path, workspace) as client:
        assert install(client, "install", export_reference(client)).status_code == 201
        shutil.rmtree(workspace / "models" / FIXTURE_PACKAGE_NAME)
        assert client.get(f"{PREFIX}/models").json() == {"items": []}
        assert client.get(f"{PREFIX}/capabilities").json()["available_models"] == []
