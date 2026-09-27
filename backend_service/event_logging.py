"""Bounded structured run logs that exclude request and exception contents."""

import json
import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

RUN_LABEL = re.compile(r"[a-z0-9_]{1,64}")


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


class SafeRotatingFileHandler(RotatingFileHandler):
    """Report log storage failures without printing a private exception chain."""

    def handleError(self, record: logging.LogRecord) -> None:
        """Replace the standard traceback fallback with fixed safe metadata."""
        try:
            print(json.dumps({"event": "event_log_write_failed"}), file=sys.stderr)
        except OSError:
            pass


def create_run_logger(
    log_directory: Path, *, label: str | None = None
) -> logging.Logger:
    """Open a bounded log file in a unique date-labelled run directory.

    A label made of lowercase letters, digits and underscores becomes part of
    the run name (``<date>_<label>_<unique>``) so the job that started the run
    can find its folder. Any other label is left out of the name.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_")
    middle = f"{label}_" if label is not None and RUN_LABEL.fullmatch(label) else ""
    run_identifier = stamp + middle + uuid4().hex
    directory = log_directory / run_identifier
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"stegolab.{run_identifier}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = SafeRotatingFileHandler(
        directory / "events.jsonl", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(EventFormatter())
    logger.addHandler(handler)
    return logger


def run_directory(logger: logging.Logger) -> Path | None:
    """Return the folder that holds this run logger's event file, if any."""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename).parent
    return None


@contextmanager
def capture_module_events(logger: logging.Logger, module_name: str) -> Iterator[None]:
    """Copy the events of one module tree into the run log while active.

    Module loggers only emit fixed event names, so the run log gains a full
    trace of the operation without any request contents.
    """
    source = logging.getLogger(module_name)
    previous_level = source.level
    handlers = list(logger.handlers)
    for handler in handlers:
        source.addHandler(handler)
    if source.getEffectiveLevel() > logging.INFO:
        source.setLevel(logging.INFO)
    try:
        yield
    finally:
        for handler in handlers:
            source.removeHandler(handler)
        source.setLevel(previous_level)


def close_run_logger(logger: logging.Logger) -> None:
    """Release log files when the backend shuts down."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
