"""Shared workspace, archive, server and document helpers for fetch tests."""

import io
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from archive_fixtures import ZipMember, build_zip
from dataset_source_fixtures import LoopbackServer, Response, test_policy
from hub_fixtures import GATED_REPOSITORY
from PIL import Image

from backend_service.dataset_sources.progress import FetchProgressSink
from backend_service.dataset_sources.transport import SecureTransport
from schemas.dataset_fetch import DatasetFetchDocument, DatasetFetchProgress

ARCHIVE_PATH = "/files/div2k.zip"
ARCHIVE_URL = f"https://127.0.0.1{ARCHIVE_PATH}"
ETAG = '"div2k-v1"'
TERMS = "https://example.test/terms"
HUB_TERMS = f"https://huggingface.co/datasets/{GATED_REPOSITORY}"
SPLITS = {"DIV2K_train_HR": "train", "DIV2K_valid_HR": "held_out"}
MEMBERS = (
    "DIV2K_train_HR/0001.png",
    "DIV2K_train_HR/0002.png",
    "DIV2K_valid_HR/0801.png",
    "DIV2K_valid_HR/0802.png",
)
FAKE_TOKEN = "hf_fake_token_that_must_never_be_logged"
FIRST_PIECE = 2000
Factory = Callable[[dict[str, str] | None], SecureTransport]
Authorizations = list[dict[str, str] | None]


@dataclass(frozen=True)
class Workspace:
    """The fixed roots one fetch works with, all under a test folder."""

    root: Path
    data_root: Path
    cache_root: Path
    output_root: Path
    logs: Path


def workspace(tmp_path: Path) -> Workspace:
    """Create the workspace folder for one test and name its roots."""
    root = tmp_path / "workspace"
    root.mkdir()
    cache_root = root / ".cache" / "stegolab" / "datasets"
    return Workspace(root, root / "data", cache_root, root / "datasets", root / "logs")


def smooth_png(seed: int) -> bytes:
    """Return a real 256x256 PNG whose gradient differs per seed."""
    gradient = Image.linear_gradient("L")
    shifted = gradient.point(lambda value: (value + seed * 40) % 256)
    buffer = io.BytesIO()
    Image.merge("RGB", (gradient, shifted, gradient.rotate(90))).save(buffer, "PNG")
    return buffer.getvalue()


def div2k_archive() -> bytes:
    """Build a DIV2K-shaped zip with two train and two validation images."""
    members = [ZipMember(name, smooth_png(index)) for index, name in enumerate(MEMBERS)]
    return build_zip(members)


def archive_routes(
    archive: bytes, *, delay: float = 0.0
) -> dict[tuple[str, str], Response]:
    """Serve the archive with pause support; ``delay`` splits the body slowly."""
    headers = {"Accept-Ranges": "bytes"}
    body = Response(body=archive, headers=headers, etag=ETAG, ranges=True)
    if delay:
        pieces = [(archive[:FIRST_PIECE], 0.0), (archive[FIRST_PIECE:], delay)]
        body = Response(headers=headers, etag=ETAG, slow_chunks=pieces)
    head = Response(body=archive, headers=headers, etag=ETAG)
    return {("HEAD", ARCHIVE_PATH): head, ("GET", ARCHIVE_PATH): body}


def factory_for(server: LoopbackServer) -> tuple[Factory, Authorizations]:
    """Reach the loopback server from a port-443 address the spec accepts."""
    calls: Authorizations = []

    def connector(address: str, host: str, port: int, timeout: float) -> socket.socket:
        """Connect to the loopback server whatever port the address named."""
        return socket.create_connection((address, server.port), timeout=timeout)

    def factory(authorization: dict[str, str] | None) -> SecureTransport:
        """Record the authorization map and build a short-timeout transport."""
        calls.append(authorization)
        return SecureTransport(
            test_policy,
            authorization=authorization,
            connector=connector,
            read_timeout=5.0,
        )

    return factory, calls


def archive_spec(**overrides: Any) -> dict[str, Any]:
    """Describe the loopback archive with DIV2K folder splits."""
    spec: dict[str, Any] = {
        "source_kind": "https_archive",
        "url": ARCHIVE_URL,
        "terms_reference": TERMS,
        "archive_splits": SPLITS,
    }
    return {**spec, **overrides}


def gated_spec() -> dict[str, Any]:
    """Describe the gated Hub repository of the hub fixtures."""
    return {
        "source_kind": "hugging_face",
        "repository": GATED_REPOSITORY,
        "terms_reference": HUB_TERMS,
    }


def fetch_document(
    space: Workspace, source: dict[str, Any], **overrides: Any
) -> DatasetFetchDocument:
    """Build the worker document a fetch job would hand to the command."""
    values: dict[str, Any] = {
        "run_label": "fetch_test",
        "source": source,
        "source_name": "div2k_sample",
        "dataset_name": "div2k",
        "data_root": str(space.data_root),
        "cache_root": str(space.cache_root),
        "output_root": str(space.output_root),
    }
    return DatasetFetchDocument.model_validate({**values, **overrides})


def sink_for(space: Workspace, name: str) -> FetchProgressSink:
    """Write progress for one run under the workspace logs."""
    return FetchProgressSink(space.logs / name)


def progress_of(sink: FetchProgressSink) -> DatasetFetchProgress:
    """Read back the progress record the sink last wrote."""
    return DatasetFetchProgress.model_validate_json(sink.path.read_bytes())


def partial_files(cache_root: Path) -> list[Path]:
    """List every partial download file under the cache."""
    return sorted(cache_root.glob("*/*.partial/*.part"))


def partial_bytes(cache_root: Path) -> int:
    """Sum the bytes of every partial download file under the cache."""
    return sum(path.stat().st_size for path in partial_files(cache_root))


def staging_is_empty(root: Path) -> bool:
    """Tell whether a writer root has no leftover staging folders."""
    staging = root / ".staging"
    return not staging.exists() or not list(staging.iterdir())
