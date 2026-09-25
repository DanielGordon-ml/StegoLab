"""FastAPI application factory for one local API process."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI

from backend_service.error_handlers import install_error_handlers
from backend_service.event_logging import close_run_logger, create_run_logger
from backend_service.failures import ApplicationFailure
from backend_service.routes import router
from backend_service.storage import StateStore

if TYPE_CHECKING:
    from backend_service.inference_jobs import InferenceServices


def inference_services(application: FastAPI) -> "InferenceServices":
    """Attach upload, model, secret, and text stores to the job supervisor."""
    from backend_service.inference_jobs import InferenceServices
    from backend_service.inference_secrets import DecodedTextStore, InferenceSecretStore

    return InferenceServices(
        files=application.state.inference_files,
        installed=application.state.installed_models,
        secrets=InferenceSecretStore(),
        texts=DecodedTextStore(),
    )


def create_application(
    data_directory: Path | None = None,
    log_directory: Path | None = None,
    workspace_root: Path | None = None,
) -> FastAPI:
    """Build the API without touching disk until its lifespan starts."""
    selected_data_directory = data_directory or Path(
        os.environ.get("STEGOLAB_DATA_DIRECTORY", ".runtime")
    )
    selected_log_directory = log_directory or Path(
        os.environ.get("STEGOLAB_LOG_DIRECTORY", "logs")
    )
    selected_workspace = workspace_root or (
        selected_data_directory.parent / "workspace"
        if data_directory is not None
        else Path(os.environ.get("STEGOLAB_WORKSPACE_ROOT", "."))
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Validate saved state before accepting requests and close run logs."""
        try:
            logger = create_run_logger(selected_log_directory)
        except OSError:
            raise RuntimeError(
                "Startup failed: check access and free space for the log directory."
            ) from None
        application.state.event_logger = logger
        try:
            try:
                application.state.store = StateStore(selected_data_directory)
                from backend_service.workspace_catalog import WorkspaceCatalog
                from backend_service.workspace_jobs import WorkspaceJobService

                application.state.workspace_catalog = WorkspaceCatalog(
                    selected_workspace, selected_data_directory
                )
                from backend_service.inference_files import InferenceFileStore
                from backend_service.model_installation import InstalledModelStore

                application.state.installed_models = InstalledModelStore(
                    selected_data_directory,
                    application.state.workspace_catalog.registry.roots,
                )
                application.state.inference_files = InferenceFileStore(
                    selected_data_directory
                )
                application.state.inference_files.sweep()
                application.state.workspace_jobs = WorkspaceJobService(
                    application.state.workspace_catalog, selected_data_directory
                )
                application.state.workspace_jobs.logger = logger
                application.state.workspace_jobs.inference = inference_services(
                    application
                )
                application.state.workspace_jobs.start()
            except ApplicationFailure:
                logger.error("application_startup_failed")
                raise RuntimeError(
                    "Startup failed: saved data is unavailable or invalid. "
                    "Check storage access and restore a valid backup if needed. "
                    "Existing saved data has been kept."
                ) from None
            logger.info("application_startup_completed")
            yield
        finally:
            if hasattr(application.state, "workspace_jobs"):
                application.state.workspace_jobs.close()
            logger.info("application_shutdown_completed")
            close_run_logger(logger)

    application = FastAPI(
        title="StegoLab",
        version="0.1.0",
        lifespan=lifespan,
        description=(
            "Local experimental training and review. "
            "Inference requires a qualified model."
        ),
    )
    application.include_router(router)
    from backend_service.image_routes import image_router
    from backend_service.inference_routes import inference_router
    from backend_service.model_routes import model_router
    from backend_service.workflow_routes import job_router
    from backend_service.workspace_routes import workspace_router

    application.include_router(model_router)
    application.include_router(image_router)
    application.include_router(inference_router)
    application.include_router(workspace_router)
    application.include_router(job_router)
    install_error_handlers(application)
    return application
