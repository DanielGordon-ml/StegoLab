"""Dataset source inspection, fetch job admission and download storage routes."""

from typing import cast

from fastapi import APIRouter, Request

from backend_service.dataset_fetch_jobs import submit_fetch
from backend_service.dataset_source_service import DatasetSourceService
from backend_service.workspace_jobs import WorkspaceJobService
from schemas.dataset_fetch import DatasetFetchRequest
from schemas.dataset_sources import DatasetInspection, DatasetInspectionRequest
from schemas.dataset_storage import DatasetStorageSummary
from schemas.errors import ErrorEnvelope
from schemas.jobs import JobSnapshot

dataset_router = APIRouter(
    prefix="/api/v1",
    responses={
        status: {"model": ErrorEnvelope}
        for status in (403, 404, 409, 422, 500, 503, 504)
    },
)


def dataset_sources(request: Request) -> DatasetSourceService:
    """Read the dataset source service created during application startup."""
    return cast(DatasetSourceService, request.app.state.dataset_sources)


def job_service(request: Request) -> WorkspaceJobService:
    """Read the one job supervisor created during application startup."""
    return cast(WorkspaceJobService, request.app.state.workspace_jobs)


@dataset_router.post(
    "/datasets/inspections",
    response_model=DatasetInspection,
    operation_id="inspect_dataset_source",
)
def inspect_dataset_source(
    payload: DatasetInspectionRequest, request: Request
) -> DatasetInspection:
    """Report what a source holds and needs before any bytes are downloaded."""
    inspection = dataset_sources(request).inspect(payload)
    request.app.state.event_logger.info("dataset_source_inspection_completed")
    return inspection


@dataset_router.post(
    "/datasets/fetch_jobs",
    response_model=JobSnapshot,
    status_code=202,
    operation_id="submit_dataset_fetch_job",
)
def submit_dataset_fetch_job(
    payload: DatasetFetchRequest, request: Request
) -> JobSnapshot:
    """Queue one download-and-prepare job; a repeated request replays it."""
    accepted = submit_fetch(job_service(request), payload)
    request.app.state.event_logger.info("dataset_fetch_job_submitted")
    return accepted


@dataset_router.get(
    "/datasets/storage",
    response_model=DatasetStorageSummary,
    operation_id="read_dataset_storage",
)
def read_dataset_storage(request: Request) -> DatasetStorageSummary:
    """Show cache, raw source and prepared dataset totals with disk headroom."""
    return dataset_sources(request).storage_summary()


@dataset_router.delete(
    "/datasets/cache/unused",
    response_model=DatasetStorageSummary,
    operation_id="remove_unused_dataset_downloads",
)
def remove_unused_dataset_downloads(request: Request) -> DatasetStorageSummary:
    """Delete cached downloads no raw folder or revision uses, then summarize."""
    summary = dataset_sources(request).remove_unused()
    request.app.state.event_logger.info("dataset_cache_cleanup_completed")
    return summary
