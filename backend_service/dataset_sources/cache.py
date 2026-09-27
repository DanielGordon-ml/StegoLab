"""Immutable, checksum-verified download cache with resumable partial files."""

import hashlib
import logging
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from backend_service.dataset_files import directory_descriptor
from backend_service.dataset_sources.cache_listing import (
    clear_unused_entries,
    partial_directories,
    summarize_cache,
)
from backend_service.dataset_sources.cache_records import (
    PART_SUFFIX,
    PARTIAL_NAME,
    CacheEntry,
    CachePartial,
    cache_basename,
    cache_directory,
    cache_failure,
    cache_kind,
    cache_lock,
    current_timestamp,
    is_regular,
    is_reserved_name,
    load_entry,
    read_partial_record,
    remove_tree,
    write_manifest_file,
    write_partial_record,
)
from backend_service.dataset_sources.plans import PlannedAsset
from backend_service.dataset_sources.transfer import DownloadOutcome, PartialFile
from backend_service.dataset_storage_ops import publish_directory
from backend_service.disk_reserve import ensure_disk_reserve
from backend_service.failures import ApplicationFailure
from schemas.dataset_cache import (
    CacheEntryManifest,
    CachePartialRecord,
    DatasetCacheSummary,
)

logger = logging.getLogger(__name__)

__all__ = ["CacheEntry", "CachePartial", "DatasetCache", "cache_failure"]
MAXIMUM_IMPORT_BYTES = 2 * 1024**3
COPY_CHUNK_BYTES = 1024**2


class DatasetCache:
    """Store downloaded files by identity so later fetches reuse them."""

    def __init__(self, root: Path) -> None:
        """Remember the root; it and its lock are created on first write."""
        self.root = Path(os.path.abspath(root))

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the exclusive cache lock while metadata changes."""
        with cache_lock(self.root):
            yield

    def find_complete(
        self, kind: str, identity: str, *, verify_bytes: bool
    ) -> CacheEntry | None:
        """Return the verified entry for an identity, or None when absent or bad."""
        directory = cache_directory(self.root, kind, identity, partial=False)
        return load_entry(directory, kind, identity, verify_bytes=verify_bytes)

    def open_partial(
        self, kind: str, identity: str, asset: PlannedAsset, *, owner: str
    ) -> CachePartial:
        """Adopt a matching partial download or start a new one for this owner."""
        directory = cache_directory(self.root, kind, identity, partial=True)
        part = directory / (cache_basename(asset.path) + PART_SUFFIX)
        with self._locked():
            existing = read_partial_record(directory)
            adoptable = (
                existing is not None
                and is_regular(part)
                and (existing.expected_sha256, existing.expected_size, existing.etag)
                == (asset.expected_sha256, asset.expected_size, asset.etag)
            )
            if not adoptable and (directory.exists() or directory.is_symlink()):
                remove_tree(directory)
                logger.info("cache_partial_discarded")
            received = part.lstat().st_size if adoptable else 0
            remaining = max((asset.expected_size or 0) - received, 0)
            ensure_disk_reserve(self.root, remaining)
            with directory_descriptor(directory, create=True):
                pass
            record = CachePartialRecord(
                owner=owner,
                bytes_received=received,
                expected_size=asset.expected_size,
                expected_sha256=asset.expected_sha256,
                etag=asset.etag,
                updated_at=current_timestamp(),
            )
            write_partial_record(directory, record)
        logger.info("cache_partial_opened")
        file = PartialFile(part)
        return CachePartial(kind, identity, directory, file, record, asset.path)

    def update_partial(self, partial: CachePartial, bytes_received: int) -> None:
        """Record how many bytes the partial file now holds."""
        fields = partial.record.model_dump(mode="json")
        fields.update(bytes_received=bytes_received, updated_at=current_timestamp())
        record = CachePartialRecord.model_validate(fields)
        with self._locked():
            write_partial_record(partial.directory, record)
        partial.record = record

    def _discard(self, partial: CachePartial, code: str) -> ApplicationFailure:
        """Remove a partial that can never complete and return the failure."""
        with self._locked():
            remove_tree(partial.directory)
        logger.warning("cache_partial_discarded")
        return cache_failure(code)

    def _adopt_existing(self, kind: str, identity: str) -> CacheEntry | None:
        """Return an entry that appeared meanwhile, refusing a damaged one."""
        directory = cache_directory(self.root, kind, identity, partial=False)
        if not directory.exists() and not directory.is_symlink():
            return None
        entry = load_entry(directory, kind, identity, verify_bytes=True)
        if entry is None:
            raise cache_failure("cache_entry_corrupt")
        logger.info("cache_entry_adopted")
        return entry

    def _seal_partial(self, partial: CachePartial, basename: str) -> None:
        """Rename the part file to its final name and drop the partial record."""
        try:
            with directory_descriptor(partial.directory) as parent:
                os.rename(
                    partial.file.path.name,
                    basename,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                )
                os.unlink(PARTIAL_NAME, dir_fd=parent)
                os.fsync(parent)
        except OSError:
            raise cache_failure("cache_storage") from None

    def complete(
        self,
        partial: CachePartial,
        outcome: DownloadOutcome,
        *,
        reference: str,
        revision: str,
        terms_reference: str,
        created_at: str,
    ) -> CacheEntry:
        """Verify the finished partial and publish it as an immutable entry."""
        partial.file.close()
        record = partial.record
        if record.expected_sha256 and outcome.sha256 != record.expected_sha256.lower():
            raise self._discard(partial, "cache_checksum_mismatch")
        if record.expected_size is not None and outcome.size != record.expected_size:
            raise self._discard(partial, "cache_size_mismatch")
        part = partial.file.path
        if not is_regular(part) or part.lstat().st_size != outcome.size:
            raise cache_failure("cache_storage")
        basename = part.name[: -len(PART_SUFFIX)]
        if is_reserved_name(basename):
            raise cache_failure("cache_key")
        manifest = CacheEntryManifest(
            kind=cache_kind(partial.kind),
            identity=partial.identity,
            reference=reference,
            revision=revision,
            path=partial.path or basename,
            basename=basename,
            size_bytes=outcome.size,
            sha256=outcome.sha256,
            etag=record.etag,
            terms_reference=terms_reference,
            created_at=created_at,
        )
        destination = cache_directory(
            self.root, partial.kind, partial.identity, partial=False
        )
        with self._locked():
            existing = self._adopt_existing(partial.kind, partial.identity)
            if existing is None:
                self._seal_partial(partial, basename)
                write_manifest_file(partial.directory, manifest)
                try:
                    publish_directory(partial.directory, destination)
                except ApplicationFailure as failure:
                    if failure.code != "dataset_exists":
                        raise
                    existing = self._adopt_existing(partial.kind, partial.identity)
            if existing is not None:
                remove_tree(partial.directory)
                return existing
        logger.info("cache_entry_published")
        file = destination / basename
        return CacheEntry(partial.kind, partial.identity, destination, file, manifest)

    def import_file(
        self,
        kind: str,
        identity: str,
        asset: PlannedAsset,
        source_file: Path,
        *,
        owner: str,
        reference: str,
        revision: str,
        terms_reference: str,
        created_at: str,
    ) -> CacheEntry:
        """Copy a local file into the cache, hashing it, then publish the entry."""
        limit = asset.expected_size
        limit = MAXIMUM_IMPORT_BYTES if limit is None else limit
        partial = self.open_partial(kind, identity, asset, owner=owner)
        digest, total = hashlib.sha256(), 0
        try:
            partial.file.open_for_append()
            partial.file.truncate()
            descriptor = os.open(source_file, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise cache_failure("cache_storage")
                while chunk := stream.read(COPY_CHUNK_BYTES):
                    total += len(chunk)
                    if total > limit:
                        raise cache_failure("cache_too_large")
                    digest.update(chunk)
                    partial.file.append(chunk)
        except OSError:
            partial.file.close()
            raise self._discard(partial, "cache_storage") from None
        except ApplicationFailure as failure:
            partial.file.close()
            raise self._discard(partial, failure.code) from None
        finally:
            partial.file.close()
        outcome = DownloadOutcome(size=total, sha256=digest.hexdigest(), resumed_from=0)
        return self.complete(
            partial,
            outcome,
            reference=reference,
            revision=revision,
            terms_reference=terms_reference,
            created_at=created_at,
        )

    def remove_partial(self, kind: str, identity: str, *, owner: str) -> bool:
        """Delete a partial download, but only on behalf of its owner."""
        directory = cache_directory(self.root, kind, identity, partial=True)
        with self._locked():
            record = read_partial_record(directory)
            if record is None or record.owner != owner:
                return False
            remove_tree(directory)
        logger.info("cache_partial_removed")
        return True

    def partial_directories(self, owner: str) -> list[str]:
        """List this owner's partial directories relative to the cache root."""
        return partial_directories(self.root, owner)

    def summary(self, in_use: dict[str, list[str]]) -> DatasetCacheSummary:
        """Describe every entry and partial, marking which entries are in use."""
        return summarize_cache(self.root, in_use)

    def clear_unused(
        self, in_use: dict[str, list[str]], *, active_owners: set[str]
    ) -> DatasetCacheSummary:
        """Delete entries nobody uses and partials whose owner is no longer active."""
        return clear_unused_entries(self.root, in_use, active_owners=active_owners)
