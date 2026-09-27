"""Read-only cache summaries and the unused-entry sweep."""

import logging
from pathlib import Path

from backend_service.dataset_sources import cache_records
from backend_service.dataset_sources.cache_records import (
    PART_SUFFIX,
    cache_lock,
    is_regular,
    read_manifest,
    read_partial_record,
    remove_tree,
    scan_cache,
)
from schemas.dataset_cache import DatasetCacheEntrySummary, DatasetCacheSummary

logger = logging.getLogger(__name__)

MAXIMUM_PARTIAL_DIRECTORIES = 8
MAXIMUM_SUMMARY_ENTRIES = cache_records.MAXIMUM_SUMMARY_ENTRIES


def partial_directories(root: Path, owner: str) -> list[str]:
    """List one owner's partial directories relative to the cache root."""
    found: list[str] = []
    for kind, name, path, is_partial in scan_cache(root):
        record = read_partial_record(path) if is_partial else None
        if record is not None and record.owner == owner:
            found.append(f"{kind}/{name}")
        if len(found) >= MAXIMUM_PARTIAL_DIRECTORIES:
            break
    return found


def _partial_bytes(path: Path) -> int:
    """Sum the bytes held by the part files of one partial directory."""
    return sum(
        item.lstat().st_size
        for item in path.iterdir()
        if item.name.endswith(PART_SUFFIX) and is_regular(item)
    )


def summarize_cache(root: Path, in_use: dict[str, list[str]]) -> DatasetCacheSummary:
    """Describe every entry and partial, marking which entries are in use.

    Every entry counts towards the totals; only the first entries, in name
    order, are listed one by one and the rest are counted as unlisted.
    """
    entries: list[DatasetCacheEntrySummary] = []
    total = unused = unused_entries = partial_bytes = partial_entries = 0
    unlisted = 0
    for _kind, _name, path, is_partial in scan_cache(root):
        if is_partial:
            partial_entries += 1
            partial_bytes += _partial_bytes(path)
            continue
        manifest = read_manifest(path)
        if manifest is None:
            logger.warning("cache_entry_invalid")
            continue
        users = list(in_use.get(manifest.identity, []))
        if len(entries) < MAXIMUM_SUMMARY_ENTRIES:
            entries.append(
                DatasetCacheEntrySummary(
                    identity=manifest.identity,
                    kind=manifest.kind,
                    reference=manifest.reference,
                    revision=manifest.revision,
                    path=manifest.path,
                    size_bytes=manifest.size_bytes,
                    created_at=manifest.created_at,
                    in_use_by=users,
                )
            )
        else:
            unlisted += 1
        total += manifest.size_bytes
        if not users:
            unused += manifest.size_bytes
            unused_entries += 1
    return DatasetCacheSummary(
        root_available=not root.is_symlink() and root.is_dir(),
        entries=entries,
        unlisted_entries=unlisted,
        total_bytes=total,
        unused_bytes=unused,
        unused_entries=unused_entries,
        partial_bytes=partial_bytes,
        partial_entries=partial_entries,
    )


def clear_unused_entries(
    root: Path, in_use: dict[str, list[str]], *, active_owners: set[str]
) -> DatasetCacheSummary:
    """Delete entries nobody uses and partials whose owner is no longer active."""
    with cache_lock(root):
        for _kind, name, path, is_partial in list(scan_cache(root)):
            if is_partial:
                record = read_partial_record(path)
                if record is None or record.owner not in active_owners:
                    remove_tree(path)
                    logger.info("cache_partial_cleared")
                continue
            manifest = read_manifest(path)
            identity = name if manifest is None else manifest.identity
            if not in_use.get(identity):
                remove_tree(path)
                logger.info("cache_entry_cleared")
    return summarize_cache(root, in_use)
