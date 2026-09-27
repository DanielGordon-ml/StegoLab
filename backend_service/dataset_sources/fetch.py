"""Fetch one dataset source end to end: resolve, download, materialize, prepare."""

import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from backend_service.dataset_commands import DatasetTerminated
from backend_service.dataset_files import directory_descriptor
from backend_service.dataset_sources import fetch_downloads
from backend_service.dataset_sources.cache import DatasetCache
from backend_service.dataset_sources.fetch_downloads import (
    DownloadRun,
    StopCheck,
    TransportFactory,
)
from backend_service.dataset_sources.fetch_reuse import reusable_source
from backend_service.dataset_sources.materialization import (
    MaterializedSource,
    find_materialized,
    materialize,
)
from backend_service.dataset_sources.plans import (
    ResolvedSource,
    inspection_for,
    plan_for,
)
from backend_service.dataset_sources.progress import FetchProgressSink
from backend_service.dataset_sources.registry import adapter_for
from backend_service.dataset_sources.source_marker import preparation_document
from backend_service.dataset_sources.transfer import DownloadStopped
from backend_service.failures import ApplicationFailure
from backend_service.hugging_face_credentials import hugging_face_token_configured
from schemas.dataset_fetch import DatasetFetchDocument, DatasetFetchSummary
from schemas.dataset_manifest import DatasetSummary
from schemas.dataset_records import DatasetPreparationRequest
from schemas.dataset_sources import MAXIMUM_IMAGES, DatasetInspection, DatasetSourceSpec

if TYPE_CHECKING:
    from backend_service.dataset_sources.upload_sessions import DatasetUploadStore

logger = logging.getLogger(__name__)

INTERRUPTIONS = (DownloadStopped, DatasetTerminated, KeyboardInterrupt)
MAXIMUM_WARNINGS = 64
TEXT_WARNING = "Text sources are saved only; no dataset revision was prepared."
_FAILURES: dict[str, tuple[str, int]] = {
    "source_access_required": (
        "This source is not accessible with the access the backend has. "
        "Inspect the source for guidance. Nothing was downloaded.",
        403,
    ),
    "source_not_found": (
        "The source was not found at its reference. Check it with the "
        "publisher. Nothing was downloaded.",
        404,
    ),
    "source_unsupported": (
        "This source cannot be fetched by this release. Nothing was downloaded.",
        422,
    ),
}
_ACCESS_CODES = {
    "access_required": "source_access_required",
    "not_found": "source_not_found",
}


def fetch_failure(code: str, guidance: str | None = None) -> ApplicationFailure:
    """Return the failure for a fetch code, preferring the adapter's guidance."""
    message, status_code = _FAILURES[code]
    return ApplicationFailure(code, guidance or message, status_code)


@dataclass(kw_only=True)
class _FetchRun(DownloadRun):
    """A download run plus the document, roots and warnings of the whole fetch."""

    document: DatasetFetchDocument
    data_root: Path
    warnings: list[str] = field(default_factory=list)


def _prepare_roots(*roots: Path) -> None:
    """Create the data and cache roots so disk checks and locks can run."""
    for root in roots:
        with directory_descriptor(root, create=True):
            pass


def _resolve(
    spec: DatasetSourceSpec,
    *,
    transport_factory: TransportFactory | None,
    uploads: "DatasetUploadStore | None",
) -> ResolvedSource:
    """Ask the adapter for this source kind what the source holds."""
    factory = transport_factory or fetch_downloads.default_transport_factory
    return adapter_for(spec).resolve(spec, transport_factory=factory, uploads=uploads)


def _require_available(resolved: ResolvedSource) -> None:
    """Refuse to continue when the source is gated, missing or unsupported."""
    if resolved.access == "available":
        return
    code = _ACCESS_CODES.get(resolved.access, "source_unsupported")
    logger.warning(code)
    raise fetch_failure(code, resolved.access_guidance)


def inspect_source(
    spec: DatasetSourceSpec,
    *,
    source_name: str | None = None,
    data_root: Path,
    cache_root: Path,
    transport_factory: TransportFactory | None = None,
    uploads: "DatasetUploadStore | None" = None,
    maximum_images: int = MAXIMUM_IMAGES,
) -> DatasetInspection:
    """Report what a fetch would download, reuse and need, without downloading."""
    _prepare_roots(data_root, cache_root)
    resolved = _resolve(spec, transport_factory=transport_factory, uploads=uploads)
    logger.info("source_inspected")
    return inspection_for(
        spec,
        resolved,
        source_name=source_name,
        cache=DatasetCache(cache_root),
        data_root=data_root,
        token_configured=hugging_face_token_configured(),
        maximum_images=maximum_images,
    )


def _prepare_phase(run: _FetchRun, result: MaterializedSource) -> DatasetSummary | None:
    """Prepare a revision from the raw folder when the document asks for one."""
    from backend_service.dataset_preparation import prepare_dataset

    document = run.document
    if not document.prepare:
        return None
    if result.marker.content != "images":
        logger.info("fetch_prepare_skipped_for_text")
        run.warnings.append(TEXT_WARNING)
        return None
    fields = preparation_document(
        result.directory, document.dataset_name, Path(document.output_root)
    )
    fields.update(training_intended=document.training_intended, seed=document.seed)
    request = DatasetPreparationRequest.model_validate(fields)
    run.progress.update(
        phase="preparing", files_completed=0, files_total=result.marker.member_count
    )
    logger.info("fetch_preparing")
    return prepare_dataset(request, progress=run.progress.prepare_event)


def _summary(
    run: _FetchRun,
    *,
    status: str,
    result: MaterializedSource | None = None,
    dataset: DatasetSummary | None = None,
    stop_signal: int | None = None,
) -> DatasetFetchSummary:
    """Describe the fetch; a reused folder reports its assets as present."""
    resolved, report = run.plan.resolved, run.report
    marker = None if result is None else result.marker
    reused = result is not None and result.reused
    assets_total = len(marker.assets) if marker and reused else len(resolved.assets)
    warnings = list(resolved.warnings)
    if result is not None:
        warnings.extend((*result.warnings, *run.warnings))
    member_count = 0 if marker is None else marker.member_count
    rejected_count = 0 if marker is None else marker.rejected_member_count
    reasons = {} if result is None else dict(result.rejection_reasons)
    return DatasetFetchSummary.model_validate(
        {
            "status": status,
            "stop_signal": stop_signal,
            "source_kind": resolved.source_kind,
            "source_name": run.document.source_name,
            "reference": resolved.reference,
            "resolved_revision": resolved.resolved_revision,
            "materialization_identity": run.plan.identity,
            "raw_folder": f"{run.data_root.name}/{run.document.source_name}",
            "bytes_received": report.bytes_received,
            "bytes_total": report.bytes_total,
            "assets_completed": assets_total if reused else report.assets_completed,
            "assets_total": assets_total,
            "member_count": member_count,
            "rejected_member_count": rejected_count,
            "rejection_reasons": reasons,
            "text_corpus": None if result is None else result.text_corpus,
            "dataset": dataset,
            "warnings": warnings[:MAXIMUM_WARNINGS],
        }
    )


def _stopped(run: _FetchRun, interruption: BaseException) -> DatasetFetchSummary:
    """Describe a fetch that stopped early; partial downloads stay for later."""
    stop_signal: int | None = None
    if isinstance(interruption, DatasetTerminated):
        stop_signal = int(signal.SIGTERM)
    elif isinstance(interruption, KeyboardInterrupt):
        stop_signal = int(signal.SIGINT)
    run.progress.update(partial_directories=run.cache.partial_directories(run.owner))
    run.progress.flush()
    logger.info("fetch_stopped")
    return _summary(run, status="stopped", stop_signal=stop_signal)


def _run_phases(run: _FetchRun) -> DatasetFetchSummary:
    """Download, materialize and prepare, reusing whatever already exists."""
    existing = find_materialized(run.data_root, run.plan)
    if existing is None:
        fetch_downloads.download_assets(run, run.data_root)
    run.progress.update(phase="extracting")
    logger.info("fetch_extracting")
    result = existing or materialize(
        run.plan,
        run.report.entries,
        data_root=run.data_root,
        cache_root=run.cache.root,
        created_at=run.created_at,
        progress=lambda done, total: run.progress.update(),
        stop=run.stop,
    )
    dataset = _prepare_phase(run, result)
    run.progress.update(phase="completed")
    run.progress.flush()
    logger.info("fetch_completed")
    status = "reused" if result.reused else "completed"
    return _summary(run, status=status, result=result, dataset=dataset)


def fetch_dataset(
    document: DatasetFetchDocument,
    *,
    progress: FetchProgressSink,
    stop: StopCheck = lambda: False,
    transport_factory: TransportFactory | None = None,
    uploads: "DatasetUploadStore | None" = None,
) -> DatasetFetchSummary:
    """Fetch the source into ``data/<source_name>/`` and prepare a revision.

    Phases: resolving, downloading (cached assets are verified and reused,
    uploads are imported, the rest streams into resumable partial files),
    extracting (the raw folder is published atomically with its marker),
    preparing (image content only, when asked) and completed. A raw folder an
    identical earlier request produced is reused offline with status
    ``reused``. A gated or missing source fails before any download. A stop
    flag, SIGTERM or keyboard interruption after resolving returns a
    ``stopped`` summary and keeps the partial downloads for the next run.
    """
    spec = document.source
    data_root = Path(os.path.abspath(document.data_root))
    cache_root = Path(os.path.abspath(document.cache_root))
    _prepare_roots(data_root, cache_root)
    progress.update(phase="resolving")
    logger.info("fetch_resolving")
    resolved = reusable_source(
        spec, data_root, document.source_name, maximum_images=document.maximum_images
    )
    if resolved is None:
        resolved = _resolve(spec, transport_factory=transport_factory, uploads=uploads)
        _require_available(resolved)
    plan = plan_for(
        spec,
        resolved,
        source_name=document.source_name,
        maximum_images=document.maximum_images,
        training_intended=document.training_intended,
    )
    run = _FetchRun(
        plan=plan,
        cache=DatasetCache(cache_root),
        owner=document.run_label,
        maximum_bytes=spec.maximum_download_bytes,
        resume=document.resume,
        created_at=datetime.now(UTC).isoformat(),
        progress=progress,
        stop=stop,
        transport_factory=transport_factory
        or fetch_downloads.default_transport_factory,
        document=document,
        data_root=data_root,
    )
    try:
        return _run_phases(run)
    except INTERRUPTIONS as interruption:
        return _stopped(run, interruption)
