"""Thin API routes shared with the local application services."""

from typing import cast

from fastapi import APIRouter, Request

from backend_service.failures import ApplicationFailure
from backend_service.hugging_face_credentials import hugging_face_token_configured
from backend_service.model_installation import InstalledModelStore
from backend_service.storage import StateStore
from backend_service.workspace_jobs import WorkspaceJobService
from schemas.capabilities import Capabilities, HealthStatus
from schemas.configuration import (
    ConfigurationProfile,
    ConfigurationReset,
    ConfigurationUpdate,
)
from schemas.errors import ErrorEnvelope
from schemas.jobs import JobList, JobSnapshot

router = APIRouter(
    prefix="/api/v1",
    responses={
        404: {"model": ErrorEnvelope},
        409: {"model": ErrorEnvelope},
        422: {"model": ErrorEnvelope},
        500: {"model": ErrorEnvelope},
        503: {"model": ErrorEnvelope},
    },
)


def state_store(request: Request) -> StateStore:
    """Read the store initialized during application startup."""
    return cast(StateStore, request.app.state.store)


def job_service(request: Request) -> WorkspaceJobService:
    """Read the persistent experimental workflow supervisor."""
    return cast(WorkspaceJobService, request.app.state.workspace_jobs)


@router.get("/health", response_model=HealthStatus, operation_id="read_health")
def read_health() -> HealthStatus:
    """Report that application startup and saved-state validation succeeded."""
    return HealthStatus()


@router.get(
    "/capabilities", response_model=Capabilities, operation_id="read_capabilities"
)
def read_capabilities(request: Request) -> Capabilities:
    """Advertise installed models and whether a server-side token exists."""
    installed = cast(InstalledModelStore, request.app.state.installed_models)
    return Capabilities.model_validate(
        {
            **installed.capabilities().model_dump(),
            "hugging_face_token_configured": hugging_face_token_configured(),
        }
    )


@router.get(
    "/configuration",
    response_model=ConfigurationProfile,
    operation_id="read_configuration",
)
def read_configuration(request: Request) -> ConfigurationProfile:
    """Read persisted settings through their strict schema."""
    return state_store(request).get_configuration()


@router.put(
    "/configuration",
    response_model=ConfigurationProfile,
    operation_id="save_configuration",
)
def save_configuration(
    payload: ConfigurationUpdate, request: Request
) -> ConfigurationProfile:
    """Save a settings update and remember its retry result atomically."""
    result = state_store(request).update_configuration(
        payload.client_request_identifier, payload.configuration
    )
    request.app.state.event_logger.info("configuration_save_completed")
    return result


@router.post(
    "/configuration/reset",
    response_model=ConfigurationProfile,
    operation_id="reset_configuration",
)
def reset_configuration(
    payload: ConfigurationReset, request: Request
) -> ConfigurationProfile:
    """Restore and persist defaults without breaking safe network retries."""
    result = state_store(request).update_configuration(
        payload.client_request_identifier, ConfigurationProfile(), operation="reset"
    )
    request.app.state.event_logger.info("configuration_reset_completed")
    return result


@router.get("/jobs", response_model=JobList, operation_id="list_jobs")
def list_jobs(request: Request) -> JobList:
    """Merge original saved records with durable workflow jobs."""
    snapshots = {item.job_identifier: item for item in state_store(request).list_jobs()}
    snapshots.update(
        {item.job_identifier: item for item in job_service(request).list_jobs()}
    )
    return JobList(items=sorted(snapshots.values(), key=lambda item: item.created_at))


@router.get(
    "/jobs/{job_identifier}", response_model=JobSnapshot, operation_id="read_job"
)
def read_job(job_identifier: str, request: Request) -> JobSnapshot:
    """Return one saved job or a safe not-found error."""
    try:
        return job_service(request).get_job(job_identifier)
    except ApplicationFailure as failure:
        if failure.code != "job_not_found":
            raise
    return state_store(request).get_job(job_identifier)
