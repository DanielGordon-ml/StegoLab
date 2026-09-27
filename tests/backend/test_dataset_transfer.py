"""Bounded, resumable and verified downloads into a partial file."""

import hashlib
from pathlib import Path

import pytest
from dataset_source_fixtures import LoopbackServer, Response, transport_for

from backend_service.dataset_sources import transfer
from backend_service.dataset_sources.transfer import (
    DownloadOutcome,
    DownloadStopped,
    PartialFile,
    RemoteAsset,
    download_asset,
)
from backend_service.failures import ApplicationFailure

BODY = bytes(range(256)) * 64
DIGEST = hashlib.sha256(BODY).hexdigest()
ETAG = '"body-v1"'
SLOW = [(BODY[:4096], 0.0), (BODY[4096:], 1.5)]


def asset(
    url: str,
    *,
    expected_size: int | None = len(BODY),
    expected_sha256: str | None = DIGEST,
    etag: str | None = ETAG,
) -> RemoteAsset:
    """Describe the test body with optional knowledge left out."""
    return RemoteAsset(
        url=url,
        basename="file.bin",
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        etag=etag,
        resumable=etag is not None,
    )


class Progress:
    """Collect every progress update in arrival order."""

    def __init__(self) -> None:
        """Start with no updates."""
        self.updates: list[tuple[int, int | None]] = []

    def update(self, *, bytes_received: int, bytes_total: int | None) -> None:
        """Record one update."""
        self.updates.append((bytes_received, bytes_total))


def test_full_download(tmp_path: Path) -> None:
    """A complete download reports its size and digest and keeps the file private."""
    routes = {("GET", "/file.bin"): Response(body=BODY, etag=ETAG)}
    partial = PartialFile(tmp_path / "file.bin.part")
    with LoopbackServer(routes) as server:
        outcome = download_asset(
            transport_for(server),
            asset(server.url("/file.bin")),
            partial,
            maximum_bytes=len(BODY),
        )
        assert "range" not in server.requests[0].headers
    assert outcome == DownloadOutcome(size=len(BODY), sha256=DIGEST, resumed_from=0)
    assert partial.path.read_bytes() == BODY
    assert partial.path.stat().st_mode & 0o777 == 0o600


def test_resume_with_matching_etag(tmp_path: Path) -> None:
    """A partial file continues from its size and the digest covers everything."""
    routes = {("GET", "/file.bin"): Response(body=BODY, etag=ETAG, ranges=True)}
    partial = PartialFile(tmp_path / "file.bin.part")
    partial.path.write_bytes(BODY[:5000])
    with LoopbackServer(routes) as server:
        outcome = download_asset(
            transport_for(server),
            asset(server.url("/file.bin")),
            partial,
            maximum_bytes=len(BODY),
        )
        request = server.requests[0]
        assert request.headers["range"] == "bytes=5000-"
        assert request.headers["if-range"] == ETAG
    assert outcome == DownloadOutcome(size=len(BODY), sha256=DIGEST, resumed_from=5000)
    assert partial.path.read_bytes() == BODY


def test_resume_restarts_when_the_etag_changed(tmp_path: Path) -> None:
    """A full 200 answer to a resume request discards the stale partial bytes."""
    routes = {("GET", "/file.bin"): Response(body=BODY, etag='"body-v2"', ranges=True)}
    partial = PartialFile(tmp_path / "file.bin.part")
    partial.path.write_bytes(b"stale" * 1000)
    with LoopbackServer(routes) as server:
        outcome = download_asset(
            transport_for(server),
            asset(server.url("/file.bin")),
            partial,
            maximum_bytes=len(BODY),
        )
        assert server.requests[0].headers["range"] == "bytes=5000-"
    assert outcome.resumed_from == 0
    assert outcome.sha256 == DIGEST
    assert partial.path.read_bytes() == BODY


def test_no_resume_without_an_etag(tmp_path: Path) -> None:
    """Without a validator the partial bytes are dropped and no Range is sent."""
    routes = {("GET", "/file.bin"): Response(body=BODY, ranges=True)}
    partial = PartialFile(tmp_path / "file.bin.part")
    partial.path.write_bytes(BODY[:100])
    with LoopbackServer(routes) as server:
        outcome = download_asset(
            transport_for(server),
            asset(server.url("/file.bin"), etag=None),
            partial,
            maximum_bytes=len(BODY),
        )
        assert "range" not in server.requests[0].headers
    assert outcome.resumed_from == 0
    assert partial.path.read_bytes() == BODY


def test_refused_status(tmp_path: Path) -> None:
    """Anything but 200 (or 206 when resuming) is a refused request."""
    routes = {("GET", "/file.bin"): Response(status=404)}
    with LoopbackServer(routes) as server:
        with pytest.raises(ApplicationFailure) as failure:
            download_asset(
                transport_for(server),
                asset(server.url("/file.bin")),
                PartialFile(tmp_path / "file.bin.part"),
                maximum_bytes=len(BODY),
            )
    assert failure.value.code == "download_status"


def test_lying_content_length(tmp_path: Path) -> None:
    """A body shorter than the declared length ends as a size mismatch."""
    routes = {
        ("GET", "/file.bin"): Response(body=BODY, lie_content_length=len(BODY) + 100)
    }
    with LoopbackServer(routes) as server:
        with pytest.raises(ApplicationFailure) as failure:
            download_asset(
                transport_for(server),
                asset(server.url("/file.bin"), expected_size=None),
                PartialFile(tmp_path / "file.bin.part"),
                maximum_bytes=len(BODY) + 100,
            )
    assert failure.value.code == "download_size_mismatch"


@pytest.mark.parametrize("chunked", [False, True])
def test_maximum_bytes(tmp_path: Path, chunked: bool) -> None:
    """The limit stops a download from the declared length or from the stream."""
    routes = {("GET", "/file.bin"): Response(body=BODY, chunked=chunked)}
    with LoopbackServer(routes) as server:
        with pytest.raises(ApplicationFailure) as failure:
            download_asset(
                transport_for(server),
                asset(server.url("/file.bin"), expected_size=None),
                PartialFile(tmp_path / "file.bin.part"),
                maximum_bytes=1000,
            )
    assert failure.value.code == "download_too_large"


def test_checksum_mismatch(tmp_path: Path) -> None:
    """A complete body with the wrong digest is reported after the transfer."""
    routes = {("GET", "/file.bin"): Response(body=BODY)}
    with LoopbackServer(routes) as server:
        with pytest.raises(ApplicationFailure) as failure:
            download_asset(
                transport_for(server),
                asset(server.url("/file.bin"), expected_sha256="0" * 64),
                PartialFile(tmp_path / "file.bin.part"),
                maximum_bytes=len(BODY),
            )
    assert failure.value.code == "download_checksum_mismatch"


def test_stop_keeps_the_partial_file(tmp_path: Path) -> None:
    """A stop request ends the stream between chunks and keeps what arrived."""
    routes = {("GET", "/file.bin"): Response(slow_chunks=SLOW)}
    partial = PartialFile(tmp_path / "file.bin.part")
    with LoopbackServer(routes) as server:
        with pytest.raises(DownloadStopped):
            download_asset(
                transport_for(server),
                asset(server.url("/file.bin")),
                partial,
                maximum_bytes=len(BODY),
                stop=lambda: partial.size > 0,
            )
    received = partial.path.read_bytes()
    assert 0 < len(received) <= 4096
    assert BODY.startswith(received)


def test_stall(tmp_path: Path) -> None:
    """No data within the stall timeout ends the download with a stall message."""
    routes = {("GET", "/file.bin"): Response(slow_chunks=SLOW)}
    with LoopbackServer(routes) as server:
        with pytest.raises(ApplicationFailure) as failure:
            download_asset(
                transport_for(server),
                asset(server.url("/file.bin")),
                PartialFile(tmp_path / "file.bin.part"),
                maximum_bytes=len(BODY),
                stall_timeout=0.5,
            )
    assert failure.value.code == "download_stalled"


def test_progress_reports_increasing_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Progress updates grow, carry the known total and end at the full size."""
    monkeypatch.setattr(transfer, "STREAM_CHUNK_BYTES", 1024)
    monkeypatch.setattr(transfer, "PROGRESS_INTERVAL_BYTES", 4096)
    routes = {("GET", "/file.bin"): Response(body=BODY)}
    progress = Progress()
    with LoopbackServer(routes) as server:
        download_asset(
            transport_for(server),
            asset(server.url("/file.bin")),
            PartialFile(tmp_path / "file.bin.part"),
            maximum_bytes=len(BODY),
            progress=progress,
        )
    counts = [received for received, _ in progress.updates]
    assert len(counts) >= 4
    assert counts == sorted(set(counts))
    assert counts[-1] == len(BODY)
    assert {total for _, total in progress.updates} == {len(BODY)}
