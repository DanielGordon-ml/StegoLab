"""Cache listings stay bounded and assets never overwrite the cache's records."""

import hashlib
from pathlib import Path

import pytest

from backend_service.dataset_sources import cache_listing
from backend_service.dataset_sources.cache import DatasetCache
from backend_service.dataset_sources.cache_records import (
    MANIFEST_NAME,
    PART_SUFFIX,
    PARTIAL_NAME,
    cache_basename,
    is_reserved_name,
)
from backend_service.dataset_sources.plans import PlannedAsset
from backend_service.dataset_sources.transfer import DownloadOutcome, PartialFile
from backend_service.failures import ApplicationFailure
from schemas.dataset_cache import CacheKind

BODY = bytes(range(256)) * 4
KIND: CacheKind = "https_archive"
PUBLISH = {
    "reference": "https://example.test/archive.zip",
    "revision": "unversioned",
    "terms_reference": "https://example.test/terms",
    "created_at": "2026-09-27T00:00:00+00:00",
}


def asset(path: str) -> PlannedAsset:
    """Describe one asset whose size and checksum are unknown beforehand."""
    return PlannedAsset(
        url="https://example.test/archive.zip",
        path=path,
        expected_size=None,
        expected_sha256=None,
        etag=None,
        resumable=False,
        authorization_host=None,
    )


def identity(index: int) -> str:
    """Derive a distinct well-formed cache identity from a number."""
    return hashlib.sha256(str(index).encode()).hexdigest()


def test_large_caches_are_listed_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thousands of entries never break partial listings, summaries or clearing."""
    cache = DatasetCache(tmp_path / "cache")
    base = tmp_path / "cache" / KIND
    base.mkdir(parents=True)
    for index in range(1001):
        (base / identity(index)).mkdir()
    partial = cache.open_partial(KIND, identity(5000), asset("a.zip"), owner="job_a")
    assert cache.partial_directories("job_a") == [f"{KIND}/{identity(5000)}.partial"]
    summary = cache.summary({})
    assert (summary.entries, summary.partial_entries) == ([], 1)
    monkeypatch.setattr(cache_listing, "MAXIMUM_SUMMARY_ENTRIES", 2)
    source = tmp_path / "upload.bin"
    source.write_bytes(BODY)
    for index in range(3):
        name = f"{index}.zip"
        cache.import_file(
            KIND, identity(6000 + index), asset(name), source, owner="job_a", **PUBLISH
        )
    listed = cache.summary({identity(6000): ["data/one"]})
    assert (len(listed.entries), listed.unlisted_entries) == (2, 1)
    assert (listed.total_bytes, listed.unused_entries) == (3 * len(BODY), 2)
    cleared = cache.clear_unused({}, active_owners=set())
    assert (cleared.entries, cleared.unlisted_entries) == ([], 0)
    assert cleared.partial_entries == 0 and not partial.directory.exists()


def test_asset_names_never_collide_with_cache_records(tmp_path: Path) -> None:
    """Reserved names get a prefix, and a reserved partial can never be sealed."""
    reserved = (
        "partial.json",
        "x/cache_manifest.json",
        "cache_manifest.json.old",
        "partial.json.tmp",
    )
    for name in reserved:
        renamed = cache_basename(name)
        assert renamed.startswith("asset_") and not is_reserved_name(renamed)
    assert cache_basename("photos.zip") == "photos.zip"
    assert cache_basename("photo.part") == "photo.part"
    cache = DatasetCache(tmp_path / "cache")
    source = tmp_path / "partial.json"
    source.write_bytes(BODY)
    entry = cache.import_file(
        "upload", identity(1), asset("partial.json"), source, owner="job_a", **PUBLISH
    )
    assert entry.file.name == "asset_partial.json" and entry.file.read_bytes() == BODY
    assert entry.manifest.path == "partial.json"
    assert (entry.directory / MANIFEST_NAME).exists()
    assert cache.find_complete("upload", identity(1), verify_bytes=True) is not None
    partial = cache.open_partial(KIND, identity(2), asset("b.zip"), owner="job_a")
    partial.file = PartialFile(partial.directory / (PARTIAL_NAME + PART_SUFFIX))
    partial.file.open_for_append()
    partial.file.append(BODY)
    partial.file.close()
    outcome = DownloadOutcome(len(BODY), hashlib.sha256(BODY).hexdigest(), 0)
    with pytest.raises(ApplicationFailure) as refused:
        cache.complete(partial, outcome, **PUBLISH)
    assert refused.value.code == "cache_key"
    assert (partial.directory / PARTIAL_NAME).exists()
