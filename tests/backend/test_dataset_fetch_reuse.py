"""Raw folders are reused only by requests with the same image cap and pin."""

import hashlib
from pathlib import Path
from typing import Any

import pytest
from archive_fixtures import ZipMember, build_zip
from dataset_source_fixtures import LoopbackServer, Response
from fetch_fixtures import (
    ARCHIVE_PATH,
    Factory,
    Workspace,
    archive_routes,
    archive_spec,
    div2k_archive,
    factory_for,
    fetch_document,
    sink_for,
    smooth_png,
    workspace,
)

from backend_service.dataset_sources.fetch import fetch_dataset, inspect_source
from backend_service.failures import ApplicationFailure
from schemas.dataset_fetch import DatasetFetchDocument, DatasetFetchSummary


@pytest.fixture(autouse=True)
def run_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep run reports inside the test folder."""
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "workspace" / "logs"))


def fetch(
    document: DatasetFetchDocument, factory: Factory, space: Workspace, name: str
) -> DatasetFetchSummary:
    """Run one fetch through the loopback transport, writing progress under a name."""
    sink = sink_for(space, name)
    return fetch_dataset(document, progress=sink, transport_factory=factory)


def test_image_cap_is_part_of_the_folder_identity(tmp_path: Path) -> None:
    """A folder capped at three images is never passed off as the full source."""
    archive, space = div2k_archive(), workspace(tmp_path)
    capped = fetch_document(space, archive_spec(), maximum_images=3, prepare=False)
    full = fetch_document(space, archive_spec(), prepare=False)
    roots: dict[str, Any] = {"data_root": space.data_root}
    roots["cache_root"] = space.cache_root
    with LoopbackServer(archive_routes(archive)) as server:
        factory, _ = factory_for(server)
        first = fetch(capped, factory, space, "capped")
        assert (first.status, first.member_count) == ("completed", 3)
        assert first.rejection_reasons == {"image_limit": 1}
        again = fetch(capped, factory, space, "capped_again")
        assert (again.status, again.member_count) == ("reused", 3)
        server.requests.clear()
        with pytest.raises(ApplicationFailure) as refused:
            fetch(full, factory, space, "full")
        assert refused.value.code == "source_folder_occupied"
        assert [request.method for request in server.requests] == ["HEAD"]
        options: dict[str, Any] = {"source_name": "div2k_sample", **roots}
        small = inspect_source(
            full.source, transport_factory=factory, maximum_images=3, **options
        )
        large = inspect_source(full.source, transport_factory=factory, **options)
        assert (small.raw_folder, large.raw_folder) == ("reusable", "conflict")
        assert small.materialization_identity == first.materialization_identity
        assert large.materialization_identity != first.materialization_identity
        renamed = fetch_document(
            space, archive_spec(), source_name="div2k_full", prepare=False
        )
        complete = fetch(renamed, factory, space, "renamed")
    assert (complete.status, complete.member_count) == ("completed", 4)
    assert (complete.bytes_received, complete.assets_completed) == (0, 1)


def test_checksum_pin_is_part_of_the_folder_identity(tmp_path: Path) -> None:
    """A request that pins a checksum never reuses a folder fetched without one."""
    archive, space = div2k_archive(), workspace(tmp_path)
    routes = {
        ("HEAD", ARCHIVE_PATH): Response(body=archive),
        ("GET", ARCHIVE_PATH): Response(body=archive),
    }
    unpinned = fetch_document(space, archive_spec(), prepare=False)
    replaced = build_zip([ZipMember("DIV2K_train_HR/0001.png", smooth_png(9))])
    digest = hashlib.sha256(replaced).hexdigest()
    pinned = fetch_document(space, archive_spec(expected_sha256=digest), prepare=False)
    with LoopbackServer(routes) as server:
        factory, _ = factory_for(server)
        first = fetch(unpinned, factory, space, "unpinned")
        assert (first.status, first.resolved_revision) == ("completed", "unversioned")
        routes[("HEAD", ARCHIVE_PATH)] = Response(body=replaced)
        routes[("GET", ARCHIVE_PATH)] = Response(body=replaced)
        server.requests.clear()
        with pytest.raises(ApplicationFailure) as refused:
            fetch(pinned, factory, space, "pinned")
        assert refused.value.code == "source_folder_occupied"
        assert [request.method for request in server.requests] == ["HEAD"]
        renamed = fetch_document(
            space,
            archive_spec(expected_sha256=digest),
            source_name="div2k_pinned",
            prepare=False,
        )
        verified = fetch(renamed, factory, space, "verified")
    assert (verified.status, verified.member_count) == ("completed", 1)
    assert verified.bytes_received == len(replaced)
    assert verified.materialization_identity != first.materialization_identity
