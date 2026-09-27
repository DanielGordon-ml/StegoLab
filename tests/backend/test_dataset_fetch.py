"""Fetch sources end to end: download, cache, raw folder, revision, stop, access."""

from pathlib import Path
from typing import Any, cast

import pytest
from dataset_source_fixtures import LoopbackServer
from fetch_fixtures import (
    ARCHIVE_PATH,
    ETAG,
    FAKE_TOKEN,
    FIRST_PIECE,
    MEMBERS,
    SPLITS,
    Factory,
    archive_routes,
    archive_spec,
    div2k_archive,
    factory_for,
    fetch_document,
    gated_spec,
    partial_bytes,
    partial_files,
    progress_of,
    sink_for,
    smooth_png,
    staging_is_empty,
    workspace,
)
from hub_fixtures import FakeHub, gated_routes

from backend_service.dataset_sources.cache import DatasetCache
from backend_service.dataset_sources.fetch import fetch_dataset, inspect_source
from backend_service.dataset_sources.fetch_reuse import REUSE_WARNING
from backend_service.dataset_sources.progress import FetchProgressSink
from backend_service.dataset_sources.transport import SecureTransport
from backend_service.dataset_sources.upload_sessions import DatasetUploadStore
from backend_service.dataset_validation import load_manifest
from backend_service.failures import ApplicationFailure
from schemas.dataset_uploads import DatasetUploadCreateRequest


@pytest.fixture(autouse=True)
def run_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep preparation run reports inside the test folder; start without a token."""
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "workspace" / "logs"))
    for variable in ("HF_TOKEN", "HF_ACCESS_TOKEN", "HF_TOKEN_FILE"):
        monkeypatch.delenv(variable, raising=False)


def test_archive_fetch_publishes_prepares_and_then_reuses_offline(
    tmp_path: Path,
) -> None:
    """One fetch yields the raw folder and a revision; the next needs no server."""
    archive, space = div2k_archive(), workspace(tmp_path)
    document = fetch_document(space, archive_spec())
    roots: dict[str, Any] = {"data_root": space.data_root}
    roots["cache_root"] = space.cache_root
    with LoopbackServer(archive_routes(archive)) as server:
        factory, calls = factory_for(server)
        options: dict[str, Any] = {"source_name": "div2k_sample", **roots}
        before = inspect_source(document.source, transport_factory=factory, **options)
        assert (before.access, before.raw_folder) == ("available", "available")
        assert (before.download_bytes, before.cached_bytes) == (len(archive), 0)
        assert before.supports_pause and before.declared_splits
        sink = sink_for(space, "first")
        summary = fetch_dataset(document, progress=sink, transport_factory=factory)
        methods = [request.method for request in server.requests]
        assert methods == ["HEAD", "HEAD", "GET"] and calls == [None] * 3
        after = inspect_source(document.source, transport_factory=factory, **options)
        assert (after.raw_folder, after.cached_bytes) == ("reusable", len(archive))
        server.requests.clear()
        again = fetch_dataset(
            document, progress=sink_for(space, "second"), transport_factory=factory
        )
        assert server.requests == [] and len(calls) == 4
    assert (summary.status, summary.stop_signal) == ("completed", None)
    assert (summary.bytes_received, summary.bytes_total) == (len(archive),) * 2
    assert (summary.assets_completed, summary.assets_total) == (1, 1)
    assert (summary.member_count, summary.rejected_member_count) == (4, 0)
    assert summary.near_duplicate_audit == "not_done" and summary.pilot_ready is False
    assert summary.raw_folder == "data/div2k_sample"
    assert summary.resolved_revision == ETAG and summary.warnings == []
    folder = space.data_root / "div2k_sample"
    for index, name in enumerate(MEMBERS):
        assert (folder / name).read_bytes() == smooth_png(index)
    assert (folder / "source-metadata.csv").exists()
    assert (folder / ".stegolab_source.json").exists()
    assert summary.dataset is not None and summary.dataset.accepted_count == 4
    assert not summary.dataset.reused
    manifest = load_manifest(space.output_root / "div2k" / summary.dataset.revision)
    expected_mapping = {"div2k_train_hr": "train", "div2k_valid_hr": "held_out"}
    assert manifest.split_mapping == expected_mapping
    provenance = manifest.source_provenance
    assert provenance["source_materialization"] == summary.materialization_identity
    assert provenance["source_name"] == "div2k_sample"
    assert provenance["source_revision"] == ETAG
    progress = progress_of(sink)
    assert progress.phase == "completed" and progress.sequence > 4
    assert (progress.files_completed, progress.files_total) == (4, 4)
    assert (progress.bytes_received, progress.assets_completed) == (len(archive), 1)
    assert progress.supports_pause and not progress.partial_directories
    assert (again.status, again.bytes_received, again.bytes_total) == ("reused", 0, 0)
    assert again.materialization_identity == summary.materialization_identity
    assert (again.member_count, again.assets_completed, again.assets_total) == (4, 1, 1)
    assert again.dataset is not None and again.dataset.reused
    assert again.dataset.revision == summary.dataset.revision
    assert again.warnings == [REUSE_WARNING]
    assert staging_is_empty(space.data_root) and staging_is_empty(space.output_root)


def test_stop_during_download_keeps_the_partial_and_a_later_run_resumes(
    tmp_path: Path,
) -> None:
    """A stop flag ends the run with the partial kept; nothing is published."""
    archive, space = div2k_archive(), workspace(tmp_path)
    routes = archive_routes(archive, delay=3.0)
    document = fetch_document(space, archive_spec())
    with LoopbackServer(routes) as server:
        factory, _ = factory_for(server)
        sink = sink_for(space, "stopped")
        stopped = fetch_dataset(
            document,
            progress=sink,
            transport_factory=factory,
            stop=lambda: partial_bytes(space.cache_root) > 0,
        )
        assert (stopped.status, stopped.stop_signal) == ("stopped", None)
        assert 0 < stopped.bytes_received <= FIRST_PIECE
        assert stopped.bytes_total == len(archive)
        assert (stopped.assets_completed, stopped.assets_total) == (0, 1)
        assert (stopped.member_count, stopped.dataset) == (0, None)
        parts = partial_files(space.cache_root)
        assert len(parts) == 1 and parts[0].stat().st_size == stopped.bytes_received
        assert not (space.data_root / "div2k_sample").exists()
        assert staging_is_empty(space.data_root)
        assert not space.output_root.exists()
        progress = progress_of(sink)
        assert progress.phase == "downloading" and progress.partial_directories
        cache = DatasetCache(space.cache_root)
        assert cache.partial_directories("fetch_test") == progress.partial_directories
        routes[("GET", ARCHIVE_PATH)] = archive_routes(archive)[("GET", ARCHIVE_PATH)]
        server.requests.clear()
        resumed = fetch_dataset(
            document, progress=sink_for(space, "resumed"), transport_factory=factory
        )
        gets = [request for request in server.requests if request.method == "GET"]
        assert gets[-1].headers["range"] == f"bytes={stopped.bytes_received}-"
        assert gets[-1].headers["if-range"] == ETAG
    assert resumed.status == "completed"
    assert resumed.bytes_received == len(archive) - stopped.bytes_received
    assert not partial_files(space.cache_root)
    assert resumed.dataset is not None and resumed.dataset.accepted_count == 4
    assert (space.data_root / "div2k_sample" / MEMBERS[0]).exists()


def test_gated_hub_source_fails_before_any_download_without_leaking_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gated repository stops at the access check with guidance and no bytes."""
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    space, hub = workspace(tmp_path), FakeHub(gated_routes())
    factory = cast(Factory, hub.transport_factory)
    document = fetch_document(
        space, gated_spec(), source_name="gated", dataset_name="gated"
    )
    sink = sink_for(space, "gated")
    with pytest.raises(ApplicationFailure) as failure:
        fetch_dataset(document, progress=sink, transport_factory=factory)
    assert failure.value.code == "source_access_required"
    assert failure.value.status_code == 403
    assert "Accept its terms" in failure.value.message
    assert [request.method for request in hub.requests] == ["GET"]
    assert "/resolve/" not in hub.requests[0].url
    assert hub.requests[0].headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert sorted(space.cache_root.rglob("*")) == []
    assert not (space.data_root / "gated").exists()
    assert progress_of(sink).phase == "resolving"
    inspection = inspect_source(
        document.source,
        data_root=space.data_root,
        cache_root=space.cache_root,
        transport_factory=factory,
    )
    assert inspection.access == "access_required" and inspection.server_token_configured
    assert inspection.access_guidance is not None and inspection.download_bytes == 0
    printed = inspection.model_dump_json()
    for text in (failure.value.message, sink.path.read_text(), printed):
        assert FAKE_TOKEN not in text


def test_uploaded_archive_is_imported_without_a_network(tmp_path: Path) -> None:
    """A completed upload becomes a cache entry and a raw folder; no transport."""
    archive, space = div2k_archive(), workspace(tmp_path)
    store = DatasetUploadStore(tmp_path / "state")
    request = DatasetUploadCreateRequest(
        client_request_identifier="fetch-upload",
        file_name="DIV2K sample.zip",
        total_bytes=len(archive),
    )
    identifier = store.create(request).upload_identifier
    store.receive_chunk(identifier, 0, archive)
    store.complete(identifier)
    source: dict[str, Any] = {"source_kind": "upload", "upload_identifier": identifier}
    source["archive_splits"] = SPLITS
    document = fetch_document(
        space, source, source_name="uploaded", dataset_name="uploaded", prepare=False
    )

    def refuse(authorization: dict[str, str] | None) -> SecureTransport:
        """Fail the test if the fetch ever asks for a network transport."""
        raise AssertionError("Uploaded archives must never open a transport.")

    sink = sink_for(space, "upload")
    summary = fetch_dataset(
        document, progress=sink, transport_factory=refuse, uploads=store
    )
    assert (summary.status, summary.bytes_received) == ("completed", 0)
    assert (summary.bytes_total, summary.assets_completed) == (0, 1)
    assert (summary.member_count, summary.dataset) == (4, None)
    assert summary.source_kind == "upload"
    assert (space.data_root / "uploaded" / MEMBERS[0]).read_bytes() == smooth_png(0)
    assert (space.data_root / "uploaded" / "source-metadata.csv").exists()
    assert list(space.cache_root.glob("upload/*/DIV2K_sample.zip"))


def test_an_occupied_folder_fails_before_any_download(tmp_path: Path) -> None:
    """A source name whose folder holds other files is refused after the probe."""
    archive, space = div2k_archive(), workspace(tmp_path)
    foreign = space.data_root / "div2k_sample"
    foreign.mkdir(parents=True)
    (foreign / "note.txt").write_text("not a fetched source")
    with LoopbackServer(archive_routes(archive)) as server:
        factory, _ = factory_for(server)
        with pytest.raises(ApplicationFailure) as failure:
            fetch_dataset(
                fetch_document(space, archive_spec()),
                progress=sink_for(space, "occupied"),
                transport_factory=factory,
            )
        assert [request.method for request in server.requests] == ["HEAD"]
    assert failure.value.code == "source_folder_occupied"
    assert not partial_files(space.cache_root)


def test_progress_sink_throttles_writes_and_maps_preparation_events(
    tmp_path: Path,
) -> None:
    """Counters always advance in memory; the file follows phases and thresholds."""
    sink = FetchProgressSink(
        tmp_path / "run", minimum_interval=60.0, minimum_bytes=1000
    )
    sink.update(phase="downloading", bytes_total=5000, assets_total=1)
    assert progress_of(sink).sequence == 1
    sink.update(bytes_received=10)
    assert sink.record.sequence == 2 and progress_of(sink).sequence == 1
    sink.update(bytes_received=1010)
    assert progress_of(sink).bytes_received == 1010
    sink.prepare_event("dataset_images_prepared_3")
    assert sink.record.files_completed == 0
    sink.update(phase="preparing", files_total=4)
    sink.prepare_event("dataset_images_prepared_3")
    sink.prepare_event("dataset_integrity_check_started")
    assert (sink.record.files_completed, progress_of(sink).files_completed) == (3, 3)
    sink.update(bytes_received=1011)
    sink.flush()
    written = progress_of(sink)
    assert (written.sequence, written.bytes_received) == (sink.record.sequence, 1011)
    assert not (tmp_path / "run" / "fetch_progress.json.tmp").exists()
