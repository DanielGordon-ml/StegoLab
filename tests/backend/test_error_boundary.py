"""Verify failures remain private before and after HTTP response completion."""

import logging
from pathlib import Path

import pytest
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from starlette.background import BackgroundTask

from backend_service.application import create_application
from backend_service.event_logging import SafeRotatingFileHandler


def test_background_failure_does_not_send_a_second_response(tmp_path: Path) -> None:
    """Keep a completed response valid when subsequent cleanup fails."""
    application = create_application(tmp_path / "data", tmp_path / "logs")

    def fail_after_response() -> None:
        """Raise private data after the response has already completed."""
        raise ValueError("private-background-sentinel")

    @application.get("/background-failure")
    def background_failure() -> PlainTextResponse:
        """Return a completed response with deliberately failing cleanup."""
        return PlainTextResponse(
            "complete", background=BackgroundTask(fail_after_response)
        )

    with TestClient(application) as client:
        response = client.get("/background-failure")
        assert response.status_code == 200
        assert response.text == "complete"
    logs = "".join(path.read_text() for path in (tmp_path / "logs").rglob("*.jsonl"))
    assert "application_request_failed" in logs
    assert "private-background-sentinel" not in logs


def test_log_storage_error_does_not_print_exception_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Avoid leaking a request failure if log writing also fails."""
    handler = SafeRotatingFileHandler(tmp_path / "test.log", maxBytes=1000)
    logger = logging.Logger("failed-storage-test")
    logger.addHandler(handler)

    def failed_rotation(record: logging.LogRecord) -> bool:
        """Simulate a log write failure while another exception is active."""
        raise OSError("private-storage-sentinel")

    monkeypatch.setattr(handler, "shouldRollover", failed_rotation)
    try:
        raise ValueError("private-request-sentinel")
    except ValueError:
        logger.error("application_request_failed")
    finally:
        handler.close()
    output = capsys.readouterr().err
    assert "event_log_write_failed" in output
    assert "private-" not in output
