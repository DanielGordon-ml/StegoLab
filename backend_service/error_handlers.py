"""Safe API failure responses and matching diagnostic log references."""

import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend_service.failures import ApplicationFailure
from schemas.errors import ApplicationError, ErrorEnvelope


async def handle_failure(request: Request, failure: Exception) -> JSONResponse:
    """Return only known safe messages and never echo exception details."""
    status_code = 500
    code = "application_failure"
    message = "The request could not be completed. Retry or check the local logs."
    if isinstance(failure, ApplicationFailure):
        status_code, code, message = (
            failure.status_code,
            failure.code,
            failure.message,
        )
    elif isinstance(failure, RequestValidationError):
        status_code, code = 422, "invalid_request"
        message = (
            "The request does not match the required format. Use a positive whole "
            "number of minutes, a supported version, and a valid request identifier."
        )
    elif isinstance(failure, HTTPException):
        status_code = failure.status_code
        code, message = (
            "request_unavailable",
            "This address or request method is unavailable.",
        )
    reference = uuid4().hex
    logger: logging.Logger = request.app.state.event_logger
    logger.error(
        "application_request_failed", extra={"diagnostic_reference": reference}
    )
    envelope = ErrorEnvelope(
        error=ApplicationError(
            code=code, message=message, diagnostic_reference=reference
        )
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump())


def install_error_handlers(application: FastAPI) -> None:
    """Replace validation and server errors with the public error envelope."""
    for failure_type in (
        ApplicationFailure,
        RequestValidationError,
        ResponseValidationError,
        HTTPException,
        Exception,
    ):
        application.add_exception_handler(failure_type, handle_failure)
    application.add_middleware(SafeFailureBoundary)


class SafeFailureBoundary:
    """Prevent unexpected request errors from reaching server traceback logging."""

    def __init__(self, app: ASGIApp) -> None:
        """Keep the wrapped application without changing normal responses."""
        self.application = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Return a safe error or finish a started response without raw failures."""
        if scope["type"] != "http":
            await self.application(scope, receive, send)
            return
        response_started = False

        async def tracked_send(message: Message) -> None:
            """Track response headers to avoid sending two responses on failure."""
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.application(scope, receive, tracked_send)
        except Exception as failure:
            response = await handle_failure(Request(scope), failure)
        else:
            return
        # Leave the exception context before sending to avoid chaining a private
        # failure into a possible connection error. Started responses are left
        # for the server to close; sending another body could itself fail.
        if not response_started:
            await response(scope, receive, send)
