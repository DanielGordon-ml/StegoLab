"""In-memory only stores for job secrets and short-lived recovered text."""

import hashlib
import hmac
import json
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from backend_service.failures import ApplicationFailure
from schemas.inference_jobs import (
    DECODED_TEXT_LIFETIME_SECONDS,
    MAXIMUM_INFERENCE_JOBS,
    DecodedText,
)

MAXIMUM_SECRET_BYTES = 4096


def queue_full_failure() -> ApplicationFailure:
    """Explain the fixed number of waiting encode and decode jobs."""
    return ApplicationFailure(
        "inference_queue_full",
        "Eight encode or decode jobs are already waiting or running. "
        "Wait for one to finish, then retry.",
        409,
    )


class InferenceSecretStore:
    """Hold submitted passwords and messages in memory until their job runs."""

    def __init__(self, capacity: int = MAXIMUM_INFERENCE_JOBS) -> None:
        """Bound the number of waiting secret sets to the queue size."""
        self.capacity = capacity
        self._entries: dict[str, dict[str, str]] = {}
        self.lock = threading.RLock()
        self._fingerprint_key = os.urandom(32)

    def digest(self, secrets: dict[str, str]) -> str:
        """Bind secrets to a retry identifier with a key that lives only in memory."""
        payload = json.dumps(secrets, sort_keys=True).encode("utf-8", "surrogatepass")
        return hmac.new(self._fingerprint_key, payload, hashlib.sha256).hexdigest()

    def put(self, job_identifier: str, secrets: dict[str, str]) -> None:
        """Keep one bounded secret set for an accepted job."""
        size = sum(
            len(key.encode("utf-8")) + len(value.encode("utf-8", "surrogatepass"))
            for key, value in secrets.items()
        )
        if size > MAXIMUM_SECRET_BYTES:
            raise ValueError("The job secrets exceed the bounded in-memory size.")
        with self.lock:
            if job_identifier not in self._entries and (
                len(self._entries) >= self.capacity
            ):
                raise queue_full_failure()
            self._entries[job_identifier] = dict(secrets)

    def pop(self, job_identifier: str) -> dict[str, str] | None:
        """Hand the secrets to the runner exactly once."""
        with self.lock:
            return self._entries.pop(job_identifier, None)

    def discard(self, job_identifier: str) -> None:
        """Forget secrets of a cancelled or abandoned job."""
        with self.lock:
            self._entries.pop(job_identifier, None)

    def __len__(self) -> int:
        """Count waiting secret sets, for tests and diagnostics only."""
        with self.lock:
            return len(self._entries)


class DecodedTextStore:
    """Keep recovered text in memory for a short time, never in the job record."""

    def __init__(
        self,
        lifetime_seconds: int = DECODED_TEXT_LIFETIME_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Bound the lifetime of texts; jobs finish one at a time, so count follows."""
        self.lifetime = timedelta(seconds=lifetime_seconds)
        self.clock = clock or (lambda: datetime.now(UTC))
        self._entries: dict[str, DecodedText] = {}
        self.lock = threading.RLock()

    def put(self, job_identifier: str, text: str) -> DecodedText:
        """Store one recovered text until its lifetime ends or it is discarded."""
        with self.lock:
            self.sweep()
            entry = DecodedText(
                job_identifier=job_identifier,
                text=text,
                byte_count=len(text.encode("utf-8")),
                expires_at=self.clock() + self.lifetime,
            )
            self._entries[job_identifier] = entry
            return entry

    def get(self, job_identifier: str) -> DecodedText | None:
        """Return an unexpired text or nothing, never partial content."""
        with self.lock:
            self.sweep()
            return self._entries.get(job_identifier)

    def discard(self, job_identifier: str) -> None:
        """Forget a text as soon as the reader is done with it."""
        with self.lock:
            self._entries.pop(job_identifier, None)

    def sweep(self) -> int:
        """Drop every expired text and count how many were removed."""
        with self.lock:
            now = self.clock()
            expired = [
                key for key, entry in self._entries.items() if entry.expires_at <= now
            ]
            for key in expired:
                del self._entries[key]
            return len(expired)
