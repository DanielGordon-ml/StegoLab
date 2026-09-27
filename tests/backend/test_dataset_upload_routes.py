"""Chunked dataset uploads through the API: parts, limits, completion and expiry."""

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from backend_service.application import create_application
from backend_service.dataset_sources.upload_sessions import DatasetUploadStore
from schemas.dataset_uploads import DATASET_UPLOAD_CHUNK_BYTES, DatasetUploadSession

PREFIX = "/api/v1/datasets/uploads"
MEBIBYTE = 1024**2
TOTAL_BYTES = DATASET_UPLOAD_CHUNK_BYTES + 4 * MEBIBYTE
ARCHIVE = b"PK\x03\x04" + (bytes(range(256)) * (TOTAL_BYTES // 256))[4:]
PARTS = (ARCHIVE[:DATASET_UPLOAD_CHUNK_BYTES], ARCHIVE[DATASET_UPLOAD_CHUNK_BYTES:])
ARCHIVE_SHA256 = hashlib.sha256(ARCHIVE).hexdigest()
OCTET_STREAM = {"Content-Type": "application/octet-stream"}


class Clock:
    """A clock the tests move forward by hand."""

    def __init__(self) -> None:
        """Start now."""
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        """Return the current test instant."""
        return self.now


@pytest.fixture
def clock() -> Clock:
    """Provide a controllable clock."""
    return Clock()


@pytest.fixture
def client(tmp_path: Path, clock: Clock) -> Iterator[TestClient]:
    """Serve an isolated API whose upload store runs on the test clock."""
    application = create_application(tmp_path / "data", tmp_path / "logs")
    with TestClient(application, raise_server_exceptions=False) as connection:
        application.state.dataset_uploads = DatasetUploadStore(tmp_path / "data", clock)
        yield connection


def create(
    client: TestClient,
    identifier: str = "req-1",
    *,
    total_bytes: int = TOTAL_BYTES,
    expected_sha256: str | None = None,
) -> str:
    """Open an upload session for the sample archive and return its identifier."""
    body: dict[str, object] = {
        "client_request_identifier": identifier,
        "file_name": "sample.zip",
        "total_bytes": total_bytes,
    }
    if expected_sha256 is not None:
        body["expected_sha256"] = expected_sha256
    response = client.post(PREFIX, json=body)
    assert response.status_code == 201
    session = DatasetUploadSession.model_validate_json(response.text)
    assert (session.chunk_count, session.received_chunks, session.complete) == (
        -(-total_bytes // DATASET_UPLOAD_CHUNK_BYTES),
        [],
        False,
    )
    return session.upload_identifier


def put_part(client: TestClient, identifier: str, index: int, data: bytes) -> Response:
    """Send one raw part."""
    response: Response = client.put(
        f"{PREFIX}/{identifier}/chunks/{index}", content=data, headers=OCTET_STREAM
    )
    return response


def complete(client: TestClient, identifier: str) -> Response:
    """Ask the server to assemble the archive."""
    body = {"client_request_identifier": f"complete-{identifier}"}
    response: Response = client.post(f"{PREFIX}/{identifier}/complete", json=body)
    return response


def failure(response: Any) -> tuple[int, str]:
    """Read the status and fixed error code of a failed response."""
    return response.status_code, str(response.json()["error"]["code"])


def received(client: TestClient, identifier: str) -> Any:
    """Read the part indexes the session has so far."""
    return client.get(f"{PREFIX}/{identifier}").json()["received_chunks"]


def test_parts_arrive_in_any_order_and_completion_fills_the_checksum(
    client: TestClient,
) -> None:
    """The last part may come first; completion verifies and reports the digest."""
    identifier = create(client, expected_sha256=ARCHIVE_SHA256)
    second = put_part(client, identifier, 1, PARTS[1])
    assert second.status_code == 200
    assert (second.json()["index"], second.json()["bytes"]) == (1, 4 * MEBIBYTE)
    assert second.json()["sha256"] == hashlib.sha256(PARTS[1]).hexdigest()
    assert second.json()["received_chunks"] == [1]
    listed = client.get(f"{PREFIX}/{identifier}").json()
    assert (listed["received_chunks"], listed["complete"], listed["sha256"]) == (
        [1],
        False,
        None,
    )
    assert put_part(client, identifier, 0, PARTS[0]).json()["received_chunks"] == [0, 1]
    finished = complete(client, identifier)
    assert finished.status_code == 200
    session = DatasetUploadSession.model_validate_json(finished.text)
    assert (session.complete, session.sha256) == (True, ARCHIVE_SHA256)
    assert client.get(f"{PREFIX}/{identifier}").json() == finished.json()
    assert complete(client, identifier).json() == finished.json()
    refused = put_part(client, identifier, 0, PARTS[0])
    assert failure(refused) == (409, "upload_already_complete")


def test_oversized_bodies_are_refused_while_streaming(client: TestClient) -> None:
    """A body past 16 MiB is refused whether or not it declares its length."""
    identifier = create(client)
    chunks = [b"\0" * 65_536] * (DATASET_UPLOAD_CHUNK_BYTES // 65_536 + 1)
    streamed = client.put(
        f"{PREFIX}/{identifier}/chunks/0",
        content=iter(chunks),
        headers={**OCTET_STREAM, "Transfer-Encoding": "chunked"},
    )
    assert failure(streamed) == (413, "upload_too_large")
    declared = put_part(client, identifier, 0, b"\0" * (DATASET_UPLOAD_CHUNK_BYTES + 1))
    assert failure(declared) == (413, "upload_too_large")
    assert received(client, identifier) == []


def test_wrong_sizes_indexes_and_unknown_sessions_are_refused(
    client: TestClient,
) -> None:
    """Each part must have its exact size and an index inside the part range."""
    identifier = create(client)
    assert failure(put_part(client, identifier, 0, PARTS[1])) == (
        422,
        "upload_chunk_size",
    )
    for index in (2, -1):
        outside = put_part(client, identifier, index, PARTS[1])
        assert failure(outside) == (422, "upload_chunk_index")
    not_a_number = client.put(f"{PREFIX}/{identifier}/chunks/zero", content=PARTS[1])
    assert failure(not_a_number) == (422, "invalid_request")
    unknown = put_part(client, "upload_" + "0" * 32, 0, PARTS[0])
    assert failure(unknown) == (404, "upload_not_found")
    assert received(client, identifier) == []


def test_completion_needs_every_part(client: TestClient) -> None:
    """Completing early names how many parts are still missing."""
    identifier = create(client)
    assert put_part(client, identifier, 1, PARTS[1]).status_code == 200
    early = complete(client, identifier)
    assert failure(early) == (409, "upload_incomplete")
    assert early.json()["error"]["message"] == "1 of 2 parts are still missing."


def test_checksum_mismatch_discards_the_session(client: TestClient) -> None:
    """A digest that differs from the declared one removes the whole upload."""
    identifier = create(client, expected_sha256="0" * 64)
    for index, part in enumerate(PARTS):
        assert put_part(client, identifier, index, part).status_code == 200
    assert failure(complete(client, identifier)) == (422, "upload_checksum_mismatch")
    assert failure(client.get(f"{PREFIX}/{identifier}")) == (404, "upload_not_found")


def test_duplicate_parts_are_idempotent_but_different_bytes_conflict(
    client: TestClient,
) -> None:
    """Re-sending the same bytes is accepted; other bytes at that index are not."""
    identifier = create(client)
    first = put_part(client, identifier, 0, PARTS[0])
    assert first.status_code == 200
    assert put_part(client, identifier, 0, PARTS[0]).json() == first.json()
    different = put_part(client, identifier, 0, b"\xff" + PARTS[0][1:])
    assert failure(different) == (409, "upload_chunk_mismatch")
    assert received(client, identifier) == [0]


def test_expired_sessions_are_gone(client: TestClient, clock: Clock) -> None:
    """A day later the session is removed on its next touch."""
    identifier = create(client)
    assert put_part(client, identifier, 1, PARTS[1]).status_code == 200
    clock.now += timedelta(hours=25)
    assert failure(client.get(f"{PREFIX}/{identifier}")) == (404, "upload_expired")
    assert failure(client.get(f"{PREFIX}/{identifier}")) == (404, "upload_not_found")


def test_discard_is_idempotent(client: TestClient) -> None:
    """Discarding removes the upload and repeating it is harmless."""
    identifier = create(client)
    assert put_part(client, identifier, 1, PARTS[1]).status_code == 200
    first = client.delete(f"{PREFIX}/{identifier}")
    assert (first.status_code, first.content) == (204, b"")
    assert failure(client.get(f"{PREFIX}/{identifier}")) == (404, "upload_not_found")
    assert client.delete(f"{PREFIX}/{identifier}").status_code == 204
    assert client.delete(f"{PREFIX}/not-an-upload").status_code == 204
