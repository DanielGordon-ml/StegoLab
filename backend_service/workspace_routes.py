"""Read-only research catalog and registered artifact download endpoints."""

from typing import cast

from fastapi import APIRouter, Request, Response

from backend_service.inference_files import InferenceFileStore
from backend_service.workspace_catalog import WorkspaceCatalog
from schemas.errors import ErrorEnvelope
from schemas.workspace import Workspace

workspace_router = APIRouter(
    prefix="/api/v1",
    responses={
        404: {"model": ErrorEnvelope},
        422: {"model": ErrorEnvelope},
        503: {"model": ErrorEnvelope},
    },
)


@workspace_router.get(
    "/workspace", response_model=Workspace, operation_id="read_workspace"
)
def read_workspace(request: Request) -> Workspace:
    """Inspect existing local datasets, runs, checkpoints, exports, and budgets."""
    return cast(WorkspaceCatalog, request.app.state.workspace_catalog).snapshot()


@workspace_router.get(
    "/artifacts/{artifact_identifier}", operation_id="download_workspace_artifact"
)
def download_artifact(artifact_identifier: str, request: Request) -> Response:
    """Return exact bytes of a package archive, an upload, or an encoded result."""
    if artifact_identifier.startswith(("image_", "encoded_")):
        store = cast(InferenceFileStore, request.app.state.inference_files)
        content, filename = store.read_image(artifact_identifier)
        media_type = "image/png"
    else:
        catalog = cast(WorkspaceCatalog, request.app.state.workspace_catalog)
        content, filename = catalog.download_export(artifact_identifier)
        media_type = "application/zip"
    return Response(
        content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
