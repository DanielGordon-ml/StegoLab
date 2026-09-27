"""Bring every planned asset into the cache: reuse it, import it or download it."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from backend_service.dataset_commands import DatasetTerminated
from backend_service.dataset_sources.address_policy import PRODUCTION_POLICY
from backend_service.dataset_sources.cache import (
    CacheEntry,
    CachePartial,
    DatasetCache,
)
from backend_service.dataset_sources.plans import (
    ARCHIVE_GROWTH_FACTOR,
    MaterializationPlan,
    PlannedAsset,
    asset_identity,
)
from backend_service.dataset_sources.progress import AssetProgress, FetchProgressSink
from backend_service.dataset_sources.transfer import (
    DownloadOutcome,
    DownloadStopped,
    RemoteAsset,
    download_asset,
    transfer_failure,
)
from backend_service.dataset_sources.transport import SecureTransport
from backend_service.disk_reserve import ensure_disk_reserve
from backend_service.failures import ApplicationFailure
from backend_service.hugging_face_credentials import read_hugging_face_token

logger = logging.getLogger(__name__)

TransportFactory = Callable[[dict[str, str] | None], SecureTransport]
StopCheck = Callable[[], bool]
INTERRUPTIONS = (DownloadStopped, DatasetTerminated, KeyboardInterrupt)
UNUSABLE_PARTIAL_CODES = frozenset(
    {"download_checksum_mismatch", "download_size_mismatch"}
)
UNREACHABLE_MESSAGE = (
    "A planned file has neither a download address nor a local copy. "
    "Inspect the source again before fetching it."
)


def default_transport_factory(
    authorization: dict[str, str] | None,
) -> SecureTransport:
    """Build the production transport: checked addresses, TLS and port 443 only."""
    return SecureTransport(PRODUCTION_POLICY, authorization=authorization)


@dataclass(frozen=True)
class AssetNeed:
    """One planned asset with its cache identity and any verified cache entry."""

    asset: PlannedAsset
    identity: str
    entry: CacheEntry | None


@dataclass
class DownloadReport:
    """What the download phase reused, imported or received.

    ``bytes_received`` counts only bytes that arrived over the network during
    this run; bytes resumed from an earlier stopped run and cached or imported
    files are not counted. ``bytes_total`` is the full size of the assets this
    run had to download, or None when one of those sizes is unknown.
    """

    entries: dict[str, CacheEntry] = field(default_factory=dict)
    bytes_received: int = 0
    bytes_total: int | None = 0
    bytes_present: int = 0
    assets_completed: int = 0


@dataclass
class DownloadRun:
    """Everything the download phase shares between its assets."""

    plan: MaterializationPlan
    cache: DatasetCache
    owner: str
    maximum_bytes: int
    resume: bool
    created_at: str
    progress: FetchProgressSink
    stop: StopCheck
    transport_factory: TransportFactory
    report: DownloadReport = field(default_factory=DownloadReport)
    transports: dict[str | None, SecureTransport] = field(default_factory=dict)

    def transport_for(self, asset: PlannedAsset) -> SecureTransport:
        """Open one transport per authorization host, only when first needed."""
        host = asset.authorization_host
        if host not in self.transports:
            token = read_hugging_face_token() if host is not None else None
            authorization = (
                None if host is None or token is None else {host: f"Bearer {token}"}
            )
            self.transports[host] = self.transport_factory(authorization)
            logger.info("download_transport_opened")
        return self.transports[host]

    def publish(self, partial: CachePartial, outcome: DownloadOutcome) -> CacheEntry:
        """Seal a finished partial into an immutable cache entry."""
        resolved = self.plan.resolved
        return self.cache.complete(
            partial,
            outcome,
            reference=resolved.reference,
            revision=resolved.resolved_revision,
            terms_reference=resolved.terms_reference,
            created_at=self.created_at,
        )


def asset_needs(plan: MaterializationPlan, cache: DatasetCache) -> list[AssetNeed]:
    """Pair every asset with its cache identity and any verified cache entry."""
    resolved = plan.resolved
    needs: list[AssetNeed] = []
    for asset in resolved.assets:
        identity = asset_identity(
            resolved.source_kind,
            resolved.reference,
            resolved.resolved_revision,
            asset.path,
            asset.expected_sha256,
        )
        entry = cache.find_complete(resolved.source_kind, identity, verify_bytes=True)
        needs.append(AssetNeed(asset, identity, entry))
    return needs


def remaining_bytes(needs: list[AssetNeed]) -> int | None:
    """Sum the sizes still to download, or None when one of them is unknown."""
    total = 0
    for need in needs:
        if need.entry is not None or need.asset.local_file is not None:
            continue
        if need.asset.expected_size is None:
            return None
        total += need.asset.expected_size
    return total


def check_space(run: DownloadRun, needs: list[AssetNeed], data_root: Path) -> None:
    """Refuse early when the download or the raw folder would break the reserve."""
    remaining = remaining_bytes(needs)
    if remaining is not None:
        if remaining > run.maximum_bytes:
            raise transfer_failure("download_too_large")
        ensure_disk_reserve(run.cache.root, remaining)
    sizes = [asset.expected_size for asset in run.plan.resolved.assets]
    if all(size is not None for size in sizes):
        archive = run.plan.resolved.source_kind != "hugging_face"
        growth = ARCHIVE_GROWTH_FACTOR if archive else 1.0
        total = sum(size for size in sizes if size is not None)
        ensure_disk_reserve(data_root, int(total * growth))


def _import(run: DownloadRun, need: AssetNeed, local_file: Path) -> CacheEntry:
    """Copy an uploaded archive into the cache without any network use."""
    resolved = run.plan.resolved
    entry = run.cache.import_file(
        resolved.source_kind,
        need.identity,
        need.asset,
        local_file,
        owner=run.owner,
        reference=resolved.reference,
        revision=resolved.resolved_revision,
        terms_reference=resolved.terms_reference,
        created_at=run.created_at,
    )
    logger.info("download_asset_imported")
    return entry


def _discard_partial_bytes(run: DownloadRun, partial: CachePartial) -> None:
    """Drop the bytes of an adopted partial when the request refuses to resume."""
    partial.file.open_for_append()
    try:
        partial.file.truncate()
    finally:
        partial.file.close()
    run.cache.update_partial(partial, 0)
    logger.info("download_partial_restarted")


def _download(run: DownloadRun, need: AssetNeed, url: str) -> CacheEntry:
    """Stream one asset into its partial file, then seal it into the cache."""
    asset, kind = need.asset, run.plan.resolved.source_kind
    partial = run.cache.open_partial(kind, need.identity, asset, owner=run.owner)
    if not run.resume and partial.file.size:
        _discard_partial_bytes(run, partial)
    started_from = partial.file.size
    remote = RemoteAsset(
        url=url,
        basename=asset.basename,
        expected_size=asset.expected_size,
        expected_sha256=asset.expected_sha256,
        etag=asset.etag,
        resumable=asset.resumable,
    )
    report = run.report
    try:
        outcome = download_asset(
            run.transport_for(asset),
            remote,
            partial.file,
            maximum_bytes=run.maximum_bytes,
            progress=AssetProgress(run.progress, report.bytes_present),
            stop=run.stop,
        )
    except INTERRUPTIONS:
        received = partial.file.size
        run.cache.update_partial(partial, received)
        report.bytes_received += max(received - started_from, 0)
        logger.info("download_partial_kept")
        raise
    except ApplicationFailure as failure:
        if failure.code in UNUSABLE_PARTIAL_CODES:
            run.cache.remove_partial(kind, need.identity, owner=run.owner)
        raise
    report.bytes_received += outcome.size - outcome.resumed_from
    report.bytes_present += outcome.size
    return run.publish(partial, outcome)


def ensure_assets(run: DownloadRun, needs: list[AssetNeed]) -> DownloadReport:
    """Make every asset a verified cache entry, downloading only what is missing.

    No transport is opened when every asset is already cached or local. A
    stop between assets or chunks raises ``DownloadStopped`` with the partial
    file kept for the next run.
    """
    report = run.report
    report.bytes_total = remaining_bytes(needs)
    run.progress.update(
        phase="downloading",
        assets_total=len(needs),
        assets_completed=0,
        bytes_received=0,
        bytes_total=report.bytes_total,
        supports_pause=run.plan.resolved.supports_pause,
    )
    for need in needs:
        if run.stop():
            logger.info("download_stopped_between_assets")
            raise DownloadStopped()
        if need.entry is not None:
            logger.info("download_cache_hit")
            entry = need.entry
        elif need.asset.local_file is not None:
            entry = _import(run, need, need.asset.local_file)
        elif need.asset.url is not None:
            entry = _download(run, need, need.asset.url)
        else:
            raise ApplicationFailure(
                "source_asset_unreachable", UNREACHABLE_MESSAGE, 502
            )
        report.entries[need.identity] = entry
        report.assets_completed += 1
        run.progress.update(assets_completed=report.assets_completed)
    logger.info("download_phase_completed")
    return report


def download_assets(run: DownloadRun, data_root: Path) -> DownloadReport:
    """Check the space estimate, then bring every asset into the cache."""
    needs = asset_needs(run.plan, run.cache)
    check_space(run, needs, data_root)
    return ensure_assets(run, needs)
