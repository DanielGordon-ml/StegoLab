"""Bounded structured run logs that exclude request and exception contents."""

import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4


class EventFormatter(logging.Formatter):
    """Serialize only explicitly approved event metadata."""

    def format(self, record: logging.LogRecord) -> str:
        """Exclude tracebacks, request paths, and untrusted log arguments."""
        return json.dumps(
            {
                "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
                "event": record.getMessage(),
                "diagnostic_reference": getattr(record, "diagnostic_reference", None),
            }
        )


def create_run_logger(log_directory: Path) -> logging.Logger:
    """Open a bounded log file in a unique date-labelled run directory."""
    run_identifier = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_") + uuid4().hex
    directory = log_directory / run_identifier
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"stegolab.{run_identifier}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(
        directory / "events.jsonl", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(EventFormatter())
    logger.addHandler(handler)
    return logger


def close_run_logger(logger: logging.Logger) -> None:
    """Release log files when the backend shuts down."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
