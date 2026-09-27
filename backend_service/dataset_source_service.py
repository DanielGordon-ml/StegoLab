"""Inspect dataset sources, summarize their storage and clear unused downloads."""

import logging
import os
import stat
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from backend_service.dataset_fetch_jobs import (
    discard_partials,
    fetch_failure,
    run_label_for,
)
from backend_service.dataset_sources.cache import DatasetCache
from backend_service.dataset_sources.fetch import inspect_source
from backend_service.dataset_sources.plans import asset_identity
from backend_service.dataset_sources.source_marker import read_source_marker
from backend_service.dataset_sources.upload_sessions import DatasetUploadStore
from backend_service.disk_reserve import free_disk_bytes, minimum_free_bytes
from backend_service.failures import ApplicationFailure
from backend_service.workspace_catalog import WorkspaceCatalog, source_folders
from backend_service.workspace_reports import discover, read_metadata
from schemas.dataset_manifest import DatasetManifest
from schemas.dataset_sources import (
    DatasetInspection,
    DatasetInspectionRequest,
    SourceMarker,
)
from schemas.dataset_storage import DatasetStorageSummary

if TYPE_CHECKING:
    from backend_service.workspace_jobs import WorkspaceJobService

logger = logging.getLogger(__name__)

INSPECTION_DEADLINE_SECONDS = 20.0
ACTIVE_FETCH_STATUSES = frozenset({"queued", "running", "paused"})
MAXIMUM_USERS_PER_ENTRY = 200
_MESSAGES: dict[str, tuple[str, int]] = {
    "inspection_timeout": (
        "The source did not answer in time. Check the address or repository "
        "and try again.",
        504,
    ),
    "cache_busy": (
        "A dataset download is queued, running, or paused. Wait for it to "
        "finish or cancel it first.",
        409,
    ),
}


def source_failure(code: str) -> ApplicationFailure:
    """Return the fixed plain-language failure for a dataset source code."""
    message, status_code = _MESSAGES[code]
    return ApplicationFailure(code, message, status_code)


def existing_ancestor(path: Path) -> Path:
    """Return the path itself or its nearest existing parent for disk figures."""
    candidate = Path(os.path.abspath(path))
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def folder_bytes(directory: Path) -> int:
    """Sum the sizes of the regular files below a folder without following links."""
    total = 0
    for parent, _directories, names in os.walk(directory):
        for name in names:
            try:
                information = os.lstat(os.path.join(parent, name))
            except OSError:
                continue
            total += information.st_size if stat.S_ISREG(information.st_mode) else 0
    return total


class DatasetSourceService:
    """Answer source and storage questions for the API without running a fetch."""

    def __init__(
        self,
        catalog: WorkspaceCatalog,
        uploads: DatasetUploadStore,
        jobs: "WorkspaceJobService",
    ) -> None:
        """Keep the workspace, the upload store and the job queue together."""
        self.catalog = catalog
        self.uploads = uploads
        self.jobs = jobs

    @property
    def data_root(self) -> Path:
        """Return the raw source folder, which the catalog creates on demand."""
        return self.catalog.data_root

    @property
    def cache_root(self) -> Path:
        """Return the download cache the fetch worker shares."""
        return self.catalog.cache_root

    def inspect(self, request: DatasetInspectionRequest) -> DatasetInspection:
        """Describe a source within the deadline; a slow source is a timeout.

        The inspection runs in a daemon thread so a source that never answers
        cannot hold the request worker or block a later shutdown.
        """
        outcomes: list[DatasetInspection | Exception] = []

        def work() -> None:
            """Run the inspection and keep its result or failure for the caller."""
            try:
                outcomes.append(
                    inspect_source(
                        request.source,
                        source_name=request.source_name,
                        data_root=self.data_root,
                        cache_root=self.cache_root,
                        uploads=self.uploads,
                        maximum_images=request.maximum_images,
                    )
                )
            except Exception as failure:
                outcomes.append(failure)

        thread = threading.Thread(target=work, name="dataset_inspection", daemon=True)
        thread.start()
        thread.join(INSPECTION_DEADLINE_SECONDS)
        if thread.is_alive() or not outcomes:
            logger.warning("dataset_source_inspection_timed_out")
            raise source_failure("inspection_timeout")
        if isinstance(outcomes[0], Exception):
            raise outcomes[0]
        logger.info("dataset_source_inspection_completed")
        return outcomes[0]

    def active_fetch_jobs(self) -> int:
        """Count fetch jobs that are queued, running or paused."""
        return sum(
            job.operation == "fetch_dataset" and job.status in ACTIVE_FETCH_STATUSES
            for job in self.jobs.list_jobs()
        )

    def storage_summary(self) -> DatasetStorageSummary:
        """Show where dataset bytes live and whether a cleanup is possible now.

        Cache totals include partial downloads, prepared totals come from the
        workspace snapshot, and the disk figures describe the volume that
        holds the cache. A cleanup is offered when unused entries or leftover
        partial downloads exist and no fetch job could still need them.
        """
        cache = DatasetCache(self.cache_root).summary(self.in_use())
        markers = self._markers()
        datasets = self.catalog.snapshot().datasets
        active = self.active_fetch_jobs()
        volume = existing_ancestor(self.cache_root)
        return DatasetStorageSummary(
            cache_bytes=cache.total_bytes + cache.partial_bytes,
            cache_entries=len(cache.entries)
            + cache.unlisted_entries
            + cache.partial_entries,
            cache_unused_bytes=cache.unused_bytes,
            cache_unused_entries=cache.unused_entries,
            raw_source_bytes=sum(
                folder_bytes(self.data_root / name) for name, _ in markers
            ),
            raw_source_folders=len(markers),
            prepared_bytes=sum(item.prepared_bytes for item in datasets),
            prepared_revisions=len(datasets),
            free_disk_bytes=free_disk_bytes(volume),
            minimum_free_bytes=minimum_free_bytes(volume),
            active_fetch_jobs=active,
            cleanup_available=(cache.unused_entries > 0 or cache.partial_entries > 0)
            and active == 0,
        )

    def remove_unused(self) -> DatasetStorageSummary:
        """Delete cached files nobody uses, unless a fetch job could need them.

        The queue lock is held only long enough to check for fetch jobs and
        to raise the cleanup flag that refuses new ones; the deletion itself
        runs without the lock so job actions and the scheduler keep working.
        """
        with self.jobs.lock:
            if self.active_fetch_jobs():
                raise source_failure("cache_busy")
            if self.jobs.cache_cleanup_in_progress:
                raise fetch_failure("dataset_cleanup_busy")
            self.jobs.cache_cleanup_in_progress = True
        try:
            DatasetCache(self.cache_root).clear_unused(
                self.in_use(), active_owners=set()
            )
        finally:
            with self.jobs.lock:
                self.jobs.cache_cleanup_in_progress = False
        logger.info("dataset_cache_unused_removed")
        return self.storage_summary()

    def in_use(self) -> dict[str, list[str]]:
        """Map cached file identities to the raw folders and revisions using them.

        A raw folder uses every asset its marker lists. A prepared revision
        whose manifest names the folder's materialization identity is listed
        as a user of the same files; unreadable manifests are skipped.

        The cache names an entry by the digest that was expected before the
        download, which is unknown for most sources, while the marker records
        the digest that was measured afterwards. Both names are claimed so a
        folder keeps its entry whichever way the fetch planned it.
        """
        users: dict[str, list[str]] = {}
        revisions = self._revision_users()
        for name, marker in self._markers():
            labels = [f"data/{name}"]
            labels.extend(revisions.get(marker.materialization_identity, []))
            for asset in marker.assets:
                for expected_sha256 in {asset.sha256, None}:
                    identity = asset_identity(
                        marker.source_kind,
                        marker.reference,
                        marker.resolved_revision,
                        asset.path,
                        expected_sha256,
                    )
                    entry = users.setdefault(identity, [])
                    entry.extend(label for label in labels if label not in entry)
                    del entry[MAXIMUM_USERS_PER_ENTRY:]
        return users

    def discard_partials(self, job_identifier: str) -> None:
        """Remove the partial downloads a cancelled fetch job owned."""
        discard_partials(self.cache_root, run_label_for(job_identifier))

    def _markers(self) -> list[tuple[str, SourceMarker]]:
        """List raw source folders that carry a readable provenance marker."""
        found: list[tuple[str, SourceMarker]] = []
        for name in source_folders(self.data_root):
            try:
                marker = read_source_marker(self.data_root / name)
            except ApplicationFailure:
                logger.warning("dataset_source_marker_skipped")
                continue
            if marker is not None:
                found.append((name, marker))
        return found

    def _revision_users(self) -> dict[str, list[str]]:
        """Group prepared revisions by the materialization identity they came from."""
        grouped: dict[str, list[str]] = {}
        for path in discover(self.catalog.root, ("datasets/*/*/manifest.json",)):
            try:
                manifest = DatasetManifest.model_validate_json(read_metadata(path))
            except (OSError, ValueError):
                continue
            identity = manifest.source_provenance.get("source_materialization")
            if identity is not None:
                label = f"datasets/{manifest.dataset_name}/{manifest.revision}"
                grouped.setdefault(identity, []).append(label)
        return grouped
