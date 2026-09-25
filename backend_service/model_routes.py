"""Explicit installation, listing, and removal of experimental model pairs."""

from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, Request, Response
from fastapi import Path as PathParameter

from backend_service.failures import ApplicationFailure
from backend_service.model_installation import InstalledModelStore, model_in_use
from backend_service.workspace_catalog import WorkspaceCatalog
from backend_service.workspace_jobs import WorkspaceJobService
from schemas.capabilities import ModelList
from schemas.errors import ErrorEnvelope
from schemas.models import InstalledModel, ModelInstallRequest

model_router = APIRouter(
    prefix="/api/v1",
    responses={
        404: {"model": ErrorEnvelope},
        409: {"model": ErrorEnvelope},
        422: {"model": ErrorEnvelope},
        503: {"model": ErrorEnvelope},
    },
)

ModelIdentifierPath = Annotated[
    str,
    PathParameter(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
    ),
]


def installed_models(request: Request) -> InstalledModelStore:
    """Read the installation store created during application startup."""
    return cast(InstalledModelStore, request.app.state.installed_models)


@model_router.get("/models", response_model=ModelList, operation_id="list_models")
def list_models(request: Request) -> ModelList:
    """List explicitly installed experimental model pairs."""
    return ModelList(items=installed_models(request).list_models())


@model_router.post(
    "/models/install",
    response_model=InstalledModel,
    status_code=201,
    operation_id="install_model",
)
def install_model(
    payload: ModelInstallRequest, request: Request, response: Response
) -> InstalledModel:
    """Verify a registered export pair and record it as an experimental model."""
    catalog = cast(WorkspaceCatalog, request.app.state.workspace_catalog)

    def locate() -> tuple[Path, Path]:
        """Resolve the registered role directory to its pair root and workspace."""
        role_directory = catalog.registry.resolve(payload.export_reference, "export")
        return role_directory.parent, catalog.root

    model, created = installed_models(request).install(payload, locate)
    if not created:
        response.status_code = 200
    request.app.state.event_logger.info("model_installation_completed")
    return model


@model_router.delete(
    "/models/{model_identifier}", status_code=204, operation_id="remove_model"
)
def remove_model(model_identifier: ModelIdentifierPath, request: Request) -> Response:
    """Forget an installed model unless an inference job still needs it."""
    jobs = cast(WorkspaceJobService, request.app.state.workspace_jobs).list_jobs()
    if model_in_use(jobs, model_identifier):
        raise ApplicationFailure(
            "model_in_use",
            "A queued or running job still uses this model. Wait for it to finish "
            "or cancel it, then remove the model.",
            409,
        )
    installed_models(request).remove(model_identifier)
    request.app.state.event_logger.info("model_removal_completed")
    return Response(status_code=204)
