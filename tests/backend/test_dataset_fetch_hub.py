"""Fetch a Hugging Face file end to end: stop, then resume with a Range request."""

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from archive_fixtures import noisy_png
from dataset_source_fixtures import LoopbackServer, Response, resolver_answering
from fetch_fixtures import (
    FIRST_PIECE,
    factory_for,
    fetch_document,
    partial_bytes,
    partial_files,
    progress_of,
    sink_for,
    workspace,
)
from hub_fixtures import file_entry

from backend_service.dataset_sources.fetch import fetch_dataset

REPOSITORY = "org/pictures"
SHA = "3" * 40
IMAGE_PATH = "images/cat.png"
API = f"/api/datasets/{REPOSITORY}"
RESOLVE_PATH = f"/datasets/{REPOSITORY}/resolve/{SHA}/{IMAGE_PATH}"


@pytest.fixture(autouse=True)
def hub_on_loopback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the Hub host to loopback, keep logs local and start without a token."""
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "workspace" / "logs"))
    for variable in ("HF_TOKEN", "HF_ACCESS_TOKEN", "HF_TOKEN_FILE"):
        monkeypatch.delenv(variable, raising=False)
    answers = resolver_answering({"huggingface.co": ["127.0.0.1"]})
    monkeypatch.setattr(socket, "getaddrinfo", answers)


def hub_spec() -> dict[str, Any]:
    """Describe the fixture repository at its default revision."""
    return {
        "source_kind": "hugging_face",
        "repository": REPOSITORY,
        "terms_reference": f"https://huggingface.co/datasets/{REPOSITORY}",
    }


def hub_routes(image: bytes, *, delay: float = 0.0) -> dict[tuple[str, str], Response]:
    """Serve the revision, the listing and one loose png that accepts ranges.

    The png route never sends an ETag: the address is pinned to a commit, so
    a resumed download must rely on that pin and on the Range answer alone.
    """
    revision = {"id": REPOSITORY, "sha": SHA, "gated": False, "siblings": []}
    listing = [file_entry(IMAGE_PATH, len(image))]
    body = Response(body=image, ranges=True)
    if delay:
        pieces = [(image[:FIRST_PIECE], 0.0), (image[FIRST_PIECE:], delay)]
        body = Response(ranges=True, slow_chunks=pieces)
    return {
        ("GET", f"{API}/revision/main"): Response(body=json.dumps(revision).encode()),
        ("GET", f"{API}/tree/{SHA}?recursive=true"): Response(
            body=json.dumps(listing).encode()
        ),
        ("GET", RESOLVE_PATH): body,
    }


def test_stopped_hub_download_resumes_with_a_range_request(tmp_path: Path) -> None:
    """A Hub file pinned to a commit resumes from its partial without an ETag."""
    image, space = noisy_png(128), workspace(tmp_path)
    assert len(image) > 2 * FIRST_PIECE
    routes = hub_routes(image, delay=3.0)
    names = {"source_name": "pictures", "dataset_name": "pictures"}
    document = fetch_document(space, hub_spec(), prepare=False, **names)
    with LoopbackServer(routes) as server:
        factory, _ = factory_for(server)
        sink = sink_for(space, "stopped")
        stopped = fetch_dataset(
            document,
            progress=sink,
            transport_factory=factory,
            stop=lambda: partial_bytes(space.cache_root) > 0,
        )
        assert (stopped.status, stopped.assets_completed) == ("stopped", 0)
        assert 0 < stopped.bytes_received <= FIRST_PIECE
        assert progress_of(sink).supports_pause
        parts = partial_files(space.cache_root)
        assert len(parts) == 1 and parts[0].stat().st_size == stopped.bytes_received
        routes[("GET", RESOLVE_PATH)] = hub_routes(image)[("GET", RESOLVE_PATH)]
        server.requests.clear()
        resumed = fetch_dataset(
            document, progress=sink_for(space, "resumed"), transport_factory=factory
        )
        requests = server.requests
        downloads = [request for request in requests if request.path == RESOLVE_PATH]
    assert len(downloads) == 1
    assert downloads[0].headers["range"] == f"bytes={stopped.bytes_received}-"
    assert "if-range" not in downloads[0].headers
    assert downloads[0].headers["host"] == "huggingface.co"
    assert (resumed.status, resumed.assets_completed) == ("completed", 1)
    assert resumed.bytes_received == len(image) - stopped.bytes_received
    assert resumed.member_count == 1 and not partial_files(space.cache_root)
    assert (space.data_root / "pictures" / IMAGE_PATH).read_bytes() == image
