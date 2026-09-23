"""FastAPI application factory for one local API process."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from backend_service.error_handlers import install_error_handlers
from backend_service.event_logging import close_run_logger, create_run_logger
from backend_service.failures import ApplicationFailure
from backend_service.routes import router
from backend_service.storage import StateStore


def create_application(
    data_directory: Path | None = None, log_directory: Path | None = None
) -> FastAPI:
    """Build the API without touching disk until its lifespan starts."""
    selected_data_directory = data_directory or Path(
        os.environ.get("STEGOLAB_DATA_DIRECTORY", ".runtime")
    )
    selected_log_directory = log_directory or Path(
        os.environ.get("STEGOLAB_LOG_DIRECTORY", "logs")
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
            logger.info("application_shutdown_completed")
            close_run_logger(logger)

    application = FastAPI(
        title="StegoLab",
        version="0.1.0",
        lifespan=lifespan,
        description="Local CPU foundation. Model features are not available yet.",
    )
    application.include_router(router)
    install_error_handlers(application)
    return application
