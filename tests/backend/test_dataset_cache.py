"""Cached downloads are verified, resumable, owned and cleared safely."""

import hashlib
import json
import shutil
from collections import namedtuple
from pathlib import Path

import pytest

from backend_service.dataset_sources import cache as cache_module
from backend_service.dataset_sources.cache import DatasetCache
from backend_service.dataset_sources.cache_records import (
    CachePartial,
    cache_basename,
    write_manifest_file,
)
from backend_service.dataset_sources.plans import PlannedAsset
from backend_service.dataset_sources.transfer import DownloadOutcome
from backend_service.dataset_storage_ops import publish_directory
from backend_service.failures import ApplicationFailure
from schemas.dataset_cache import CacheEntryManifest, CacheKind

Usage = namedtuple("Usage", "total used free")
BODY = bytes(range(256)) * 8
DIGEST = hashlib.sha256(BODY).hexdigest()
IDENTITY = "a" * 64
OTHER_IDENTITY = "b" * 64
KIND: CacheKind = "https_archive"
PUBLISH = {
    "reference": "https://example.test/archive.zip",
    "revision": '"etag-1"',
    "terms_reference": "https://example.test/terms",
    "created_at": "2026-09-27T00:00:00+00:00",
}


def asset(
    path: str = "archive.zip",
    *,
    size: int | None = len(BODY),
    sha256: str | None = DIGEST,
    etag: str | None = '"etag-1"',
) -> PlannedAsset:
    """Describe the test body with optional knowledge left out."""
    return PlannedAsset(
        url="https://example.test/archive.zip",
        path=path,
        expected_size=size,
        expected_sha256=sha256,
        etag=etag,
        resumable=True,
        authorization_host=None,
    )


def feed(partial: CachePartial, data: bytes) -> DownloadOutcome:
    """Append bytes to a partial as a download would and describe the result."""
    resumed_from = partial.file.open_for_append()
    partial.file.append(data)
    partial.file.close()
    whole = partial.file.path.read_bytes()
    return DownloadOutcome(len(whole), hashlib.sha256(whole).hexdigest(), resumed_from)


def test_partial_completes_into_an_immutable_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bytes flow into a partial, then the manifest is written last and published."""
    cache = DatasetCache(tmp_path / "cache")
    partial = cache.open_partial(KIND, IDENTITY, asset(), owner="job_a")
    assert partial.directory == tmp_path / "cache" / KIND / f"{IDENTITY}.partial"
    assert partial.file.path.name == "archive.zip.part"
    cache.update_partial(partial, 3)
    record = json.loads((partial.directory / "partial.json").read_text())
    assert (record["owner"], record["bytes_received"]) == ("job_a", 3)
    outcome = feed(partial, BODY)
    seen: list[tuple[bool, bool, int]] = []

    def observe(directory: Path, manifest: CacheEntryManifest) -> None:
        """Record what the directory holds at the moment the manifest is written."""
        sealed = directory / "archive.zip"
        seen.append(
            (
                sealed.exists(),
                (directory / "partial.json").exists(),
                sealed.stat().st_size,
            )
        )
        write_manifest_file(directory, manifest)

    monkeypatch.setattr(cache_module, "write_manifest_file", observe)
    entry = cache.complete(partial, outcome, **PUBLISH)
    assert seen == [(True, False, len(BODY))]
    assert entry.directory == tmp_path / "cache" / KIND / IDENTITY
    assert entry.file.read_bytes() == BODY
    assert not partial.directory.exists()
    stored = CacheEntryManifest.model_validate_json(
        (entry.directory / "cache_manifest.json").read_bytes()
    )
    assert stored == entry.manifest
    assert (stored.path, stored.basename, stored.sha256) == (
        "archive.zip",
        "archive.zip",
        DIGEST,
    )
    found = cache.find_complete(KIND, IDENTITY, verify_bytes=True)
    assert found is not None and found.manifest == stored
    assert cache.partial_directories("job_a") == []


def test_checksum_or_size_mismatch_removes_the_partial(tmp_path: Path) -> None:
    """A finished download that does not match what was promised is dropped."""
    cache = DatasetCache(tmp_path / "cache")
    partial = cache.open_partial(KIND, IDENTITY, asset(), owner="job_a")
    outcome = feed(partial, BODY[:-1] + b"\x00")
    with pytest.raises(ApplicationFailure) as failure:
        cache.complete(partial, outcome, **PUBLISH)
    assert failure.value.code == "cache_checksum_mismatch"
    assert not partial.directory.exists()
    partial = cache.open_partial(KIND, IDENTITY, asset(sha256=None), owner="job_a")
    with pytest.raises(ApplicationFailure) as short:
        cache.complete(partial, feed(partial, BODY[:10]), **PUBLISH)
    assert short.value.code == "cache_size_mismatch"
    assert not partial.directory.exists()
    assert cache.find_complete(KIND, IDENTITY, verify_bytes=False) is None


def test_verify_bytes_detects_a_tampered_file(tmp_path: Path) -> None:
    """A same-size edit passes the cheap check and fails the hashed one."""
    cache = DatasetCache(tmp_path / "cache")
    partial = cache.open_partial(KIND, IDENTITY, asset(), owner="job_a")
    entry = cache.complete(partial, feed(partial, BODY), **PUBLISH)
    entry.file.write_bytes(b"\xff" + BODY[1:])
    assert cache.find_complete(KIND, IDENTITY, verify_bytes=False) is not None
    assert cache.find_complete(KIND, IDENTITY, verify_bytes=True) is None
    entry.file.write_bytes(BODY[:5])
    assert cache.find_complete(KIND, IDENTITY, verify_bytes=False) is None


def test_matching_partial_is_adopted_and_a_foreign_one_discarded(
    tmp_path: Path,
) -> None:
    """Only a partial that promises the same bytes is resumed."""
    cache = DatasetCache(tmp_path / "cache")
    first = cache.open_partial(KIND, IDENTITY, asset(), owner="job_a")
    feed(first, BODY[:100])
    adopted = cache.open_partial(KIND, IDENTITY, asset(), owner="job_b")
    assert adopted.record.bytes_received == 100
    assert adopted.record.owner == "job_b"
    assert adopted.file.path.read_bytes() == BODY[:100]
    foreign = cache.open_partial(KIND, IDENTITY, asset(etag='"etag-2"'), owner="job_b")
    assert foreign.record.bytes_received == 0
    assert not foreign.file.path.exists()
    (foreign.directory / "partial.json").write_text("{broken")
    fresh = cache.open_partial(KIND, IDENTITY, asset(etag='"etag-2"'), owner="job_c")
    assert fresh.record.bytes_received == 0


def test_remove_partial_and_listing_respect_the_owner(tmp_path: Path) -> None:
    """Partials are listed for and removable by their owner only."""
    cache = DatasetCache(tmp_path / "cache")
    partial = cache.open_partial(KIND, IDENTITY, asset(), owner="job_a")
    assert cache.partial_directories("job_a") == [f"{KIND}/{IDENTITY}.partial"]
    assert cache.partial_directories("job_b") == []
    assert cache.remove_partial(KIND, IDENTITY, owner="job_b") is False
    assert partial.directory.exists()
    assert cache.remove_partial(KIND, IDENTITY, owner="job_a") is True
    assert not partial.directory.exists()
    assert cache.remove_partial(KIND, IDENTITY, owner="job_a") is False


def test_summary_and_clear_unused_respect_users_and_owners(tmp_path: Path) -> None:
    """Unused entries and orphaned partials go; everything referenced stays."""
    cache = DatasetCache(tmp_path / "cache")
    for identity, path in ((IDENTITY, "a.zip"), (OTHER_IDENTITY, "b.zip")):
        partial = cache.open_partial(KIND, identity, asset(path), owner="job_a")
        cache.complete(partial, feed(partial, BODY), **PUBLISH)
    kept = cache.open_partial("upload", IDENTITY, asset("c.zip"), owner="job_a")
    dropped = cache.open_partial(
        "hugging_face", IDENTITY, asset("d.zip"), owner="job_b"
    )
    feed(dropped, BODY[:64])
    in_use = {IDENTITY: ["data/div2k"]}
    summary = cache.summary(in_use)
    assert [entry.identity for entry in summary.entries] == [IDENTITY, OTHER_IDENTITY]
    assert summary.entries[0].in_use_by == ["data/div2k"]
    assert (summary.total_bytes, summary.unused_bytes) == (2 * len(BODY), len(BODY))
    assert (summary.unused_entries, summary.partial_entries) == (1, 2)
    assert summary.partial_bytes == 64
    cleared = cache.clear_unused(in_use, active_owners={"job_a"})
    assert [entry.identity for entry in cleared.entries] == [IDENTITY]
    assert (cleared.unused_entries, cleared.partial_entries) == (0, 1)
    assert kept.directory.exists() and not dropped.directory.exists()
    assert DatasetCache(tmp_path / "missing").summary({}).root_available is False


def test_entry_that_appeared_concurrently_is_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second job publishing the same identity wins and the partial is dropped."""
    cache = DatasetCache(tmp_path / "cache")
    partial = cache.open_partial(KIND, IDENTITY, asset(), owner="job_a")
    outcome = feed(partial, BODY)
    rival = tmp_path / "rival"
    rival.mkdir()
    (rival / "archive.zip").write_bytes(BODY)
    rival_manifest = CacheEntryManifest(
        kind=KIND,
        identity=IDENTITY,
        path="archive.zip",
        basename="archive.zip",
        size_bytes=len(BODY),
        sha256=DIGEST,
        etag='"etag-1"',
        reference=PUBLISH["reference"],
        revision=PUBLISH["revision"],
        terms_reference=PUBLISH["terms_reference"],
        created_at="2026-01-01T00:00:00+00:00",
    )
    write_manifest_file(rival, rival_manifest)

    def race(source: Path, destination: Path) -> None:
        """Slip the rival copy into place so the real publish finds it occupied."""
        rival.rename(destination)
        publish_directory(source, destination)

    monkeypatch.setattr(cache_module, "publish_directory", race)
    entry = cache.complete(partial, outcome, **PUBLISH)
    assert entry.manifest == rival_manifest and entry.file.read_bytes() == BODY
    assert not partial.directory.exists()
    monkeypatch.undo()
    partial = cache.open_partial(KIND, "c" * 64, asset(), owner="job_a")
    outcome = feed(partial, BODY)
    corrupt = tmp_path / "cache" / KIND / ("c" * 64)
    corrupt.mkdir()
    (corrupt / "cache_manifest.json").write_text("{}")
    with pytest.raises(ApplicationFailure) as failure:
        cache.complete(partial, outcome, **PUBLISH)
    assert failure.value.code == "cache_entry_corrupt"
    assert partial.file.path.exists()


def test_disk_reserve_blocks_a_new_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial is never opened when the download would eat the reserve."""
    cache = DatasetCache(tmp_path / "cache")
    usage = Usage(total=100 * 1024**3, used=99 * 1024**3, free=1024**3)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: usage)
    with pytest.raises(ApplicationFailure) as failure:
        cache.open_partial(KIND, IDENTITY, asset(), owner="job_a")
    assert failure.value.code == "dataset_space"
    assert not (tmp_path / "cache" / KIND).exists()


def test_import_file_copies_hashes_and_bounds(tmp_path: Path) -> None:
    """Local files are imported through a partial with the same checks."""
    cache = DatasetCache(tmp_path / "cache")
    source = tmp_path / "upload.bin"
    source.write_bytes(BODY)
    entry = cache.import_file(
        "upload", IDENTITY, asset("photos.zip"), source, owner="job_a", **PUBLISH
    )
    assert entry.file.read_bytes() == BODY
    assert (entry.manifest.kind, entry.manifest.sha256) == ("upload", DIGEST)
    with pytest.raises(ApplicationFailure) as failure:
        cache.import_file(
            "upload", OTHER_IDENTITY, asset(size=10), source, owner="job_a", **PUBLISH
        )
    assert failure.value.code == "cache_too_large"
    assert not (tmp_path / "cache" / "upload" / f"{OTHER_IDENTITY}.partial").exists()
    link = tmp_path / "link.bin"
    link.symlink_to(source)
    with pytest.raises(ApplicationFailure) as linked:
        cache.import_file(
            "upload", OTHER_IDENTITY, asset(), link, owner="job_a", **PUBLISH
        )
    assert linked.value.code == "cache_storage"


def test_keys_and_basenames_are_checked(tmp_path: Path) -> None:
    """Unknown kinds, malformed identities and odd names never reach the disk."""
    cache = DatasetCache(tmp_path / "cache")
    for kind, identity in (("local", IDENTITY), (KIND, "abc"), (KIND, "A" * 64)):
        with pytest.raises(ApplicationFailure) as failure:
            cache.find_complete(kind, identity, verify_bytes=False)
        assert failure.value.code == "cache_key"
    assert cache_basename("data/train-00000.parquet") == "train-00000.parquet"
    assert cache_basename("weird name!.zip") == "weird_name_.zip"
    assert cache_basename("...") == "asset"
