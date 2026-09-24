"""Read-only research catalog and registered artifact download endpoints."""

from typing import cast

from fastapi import APIRouter, Request, Response

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
    """Return exact verified package members as one browser-downloadable archive."""
    catalog = cast(WorkspaceCatalog, request.app.state.workspace_catalog)
    content, filename = catalog.download_export(artifact_identifier)
    return Response(
        content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
