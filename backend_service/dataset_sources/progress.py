"""Throttled progress records that a fetch writes beside its run log."""

import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from backend_service.dataset_files import directory_descriptor
from backend_service.dataset_serialization import canonical_json
from backend_service.failures import ApplicationFailure
from schemas.dataset_fetch import DatasetFetchProgress

logger = logging.getLogger(__name__)

PROGRESS_NAME = "fetch_progress.json"
MINIMUM_INTERVAL_SECONDS = 0.5
MINIMUM_BYTES = 4 * 1024**2
PREPARED_EVENT = re.compile(r"dataset_images_prepared_(\d{1,7})")
PREPARATION_FINISHED = frozenset(
    {"dataset_preparation_completed", "dataset_revision_reused"}
)
COUNTER_FIELDS = (
    "files_completed",
    "files_total",
    "assets_completed",
    "assets_total",
    "bytes_total",
    "supports_pause",
    "partial_directories",
)
_MESSAGES = {
    "fetch_progress_storage": (
        "The fetch progress file could not be written. Check access and free "
        "space for the log directory."
    ),
}


def progress_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for a progress failure code."""
    return ApplicationFailure(code, _MESSAGES[code], 503)


def current_timestamp() -> str:
    """Stamp progress records with the current UTC time."""
    return datetime.now(UTC).isoformat()


class FetchProgressSink:
    """Keep the latest fetch counters and write them atomically, not too often.

    Every ``update`` raises the sequence number. The file is rewritten when the
    phase or a file or asset counter changes, when enough bytes arrived since
    the last write, or when enough time passed; ``flush`` writes at once.
    ``bytes_received`` counts the bytes on disk for the assets this run
    downloads, including bytes resumed from an earlier stopped run, so a
    progress bar can compare it with ``bytes_total``.
    """

    def __init__(
        self,
        directory: Path,
        *,
        minimum_interval: float = MINIMUM_INTERVAL_SECONDS,
        minimum_bytes: int = MINIMUM_BYTES,
    ) -> None:
        """Remember where to write; the first update creates the file."""
        self.directory = Path(os.path.abspath(directory))
        self.path = self.directory / PROGRESS_NAME
        self.minimum_interval = minimum_interval
        self.minimum_bytes = minimum_bytes
        self.record = DatasetFetchProgress(
            sequence=0, phase="resolving", updated_at=current_timestamp()
        )
        self._written_bytes = 0
        self._written_at: float | None = None

    def update(self, **fields: object) -> None:
        """Merge new counter values, raise the sequence and write when due."""
        previous = self.record
        values = previous.model_dump(mode="json")
        values.update(fields)
        values.update(sequence=previous.sequence + 1, updated_at=current_timestamp())
        self.record = DatasetFetchProgress.model_validate(values)
        if self._due(previous):
            self.flush()

    def _due(self, previous: DatasetFetchProgress) -> bool:
        """Decide whether the record changed enough to reach the disk now."""
        record = self.record
        if self._written_at is None or record.phase != previous.phase:
            return True
        if any(
            getattr(record, name) != getattr(previous, name) for name in COUNTER_FIELDS
        ):
            return True
        if record.bytes_received - self._written_bytes >= self.minimum_bytes:
            return True
        return time.monotonic() - self._written_at >= self.minimum_interval

    def flush(self) -> None:
        """Write the current record through a temporary file and a rename."""
        data = canonical_json(self.record.model_dump(mode="json"))
        temporary = PROGRESS_NAME + ".tmp"
        try:
            with directory_descriptor(self.directory, create=True) as parent:
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
                descriptor = os.open(temporary, flags, 0o600, dir_fd=parent)
                with os.fdopen(descriptor, "wb") as output:
                    output.write(data)
                    os.fsync(output.fileno())
                os.replace(
                    temporary, PROGRESS_NAME, src_dir_fd=parent, dst_dir_fd=parent
                )
                os.fsync(parent)
        except (OSError, ApplicationFailure):
            logger.warning("fetch_progress_write_failed")
            raise progress_failure("fetch_progress_storage") from None
        self._written_bytes = self.record.bytes_received
        self._written_at = time.monotonic()

    def prepare_event(self, event: str) -> None:
        """Turn the preparation's image counter events into file progress.

        The counter events arrive every hundred images; the completion events
        settle the count at the known total. Other preparation events are
        recorded as plain event names.
        """
        match = PREPARED_EVENT.fullmatch(event)
        preparing = self.record.phase == "preparing"
        if match is not None:
            if preparing:
                self.update(files_completed=int(match.group(1)))
            return
        logger.info(event)
        total = self.record.files_total
        if preparing and event in PREPARATION_FINISHED and total is not None:
            self.update(files_completed=total)


class AssetProgress:
    """Offset one asset's byte counts by the assets downloaded before it.

    This is the ``ProgressSink`` one download reports to; the phase total was
    fixed when the download phase began, so only the running count moves.
    """

    def __init__(self, sink: FetchProgressSink, base: int) -> None:
        """Remember the sink and the bytes already present before this asset."""
        self._sink = sink
        self._base = base

    def update(self, *, bytes_received: int, bytes_total: int | None) -> None:
        """Forward the running count of this asset on top of the earlier ones."""
        self._sink.update(bytes_received=self._base + bytes_received)
