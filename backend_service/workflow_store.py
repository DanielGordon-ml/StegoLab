"""Durable local workflow snapshots, ordered events, and mutation replay."""

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

from pydantic import Field, TypeAdapter

from backend_service.failures import ApplicationFailure, StorageFailure
from schemas.inference_jobs import InferenceJobRecord
from schemas.jobs import JobEvent, JobSnapshot
from schemas.workflows import WorkflowRequest

StoredRequest = WorkflowRequest | InferenceJobRecord
REQUEST_ADAPTER: TypeAdapter[StoredRequest] = TypeAdapter(
    Annotated[StoredRequest, Field(discriminator="operation")]
)


class WorkflowStore:
    """Store public workflow metadata separately from original experiment files."""

    def __init__(self, directory: Path) -> None:
        """Create an independent catalogue database without changing the ledger."""
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "workflows.sqlite3"
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS workflow_jobs "
                "(identifier TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                "request TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS workflow_events "
                "(identifier INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS workflow_mutations "
                "(identifier TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
                "job TEXT NOT NULL)"
            )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Commit short transactions and close connections on every outcome."""
        connection = None
        try:
            connection = sqlite3.connect(self.path, timeout=5)
            with connection:
                yield connection
        except sqlite3.Error:
            raise StorageFailure() from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def fingerprint(value: object) -> str:
        """Bind one retry identifier to an exact operation and validated payload."""
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def replay(self, identifier: str, fingerprint: str) -> JobSnapshot | None:
        """Return the current accepted job while rejecting conflicting retries."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT fingerprint, job FROM workflow_mutations WHERE identifier=?",
                (identifier,),
            ).fetchone()
        if row is None:
            return None
        if row[0] != fingerprint:
            raise ApplicationFailure(
                "request_identifier_conflict",
                "This request identifier was used for another change. "
                "Refresh and retry.",
                409,
            )
        return self.get(str(row[1]))

    def get(self, identifier: str) -> JobSnapshot:
        """Read a validated saved job without inspecting experiment state."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT payload FROM workflow_jobs WHERE identifier=?", (identifier,)
            ).fetchone()
        if row is None:
            raise ApplicationFailure("job_not_found", "No saved job was found.", 404)
        return JobSnapshot.model_validate_json(row[0])

    def request(self, identifier: str) -> StoredRequest:
        """Restore the frozen public request or inference record for a job."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT request FROM workflow_jobs WHERE identifier=?", (identifier,)
            ).fetchone()
        if row is None:
            raise ApplicationFailure("job_not_found", "No saved job was found.", 404)
        return REQUEST_ADAPTER.validate_json(row[0])

    def list_jobs(self) -> list[JobSnapshot]:
        """Return stable creation order for both queue and browser history."""
        with self.connection() as connection:
            rows = connection.execute("SELECT payload FROM workflow_jobs").fetchall()
        return sorted(
            (JobSnapshot.model_validate_json(row[0]) for row in rows),
            key=lambda item: (item.created_at, item.job_identifier),
        )

    def save(
        self,
        snapshot: JobSnapshot,
        *,
        request: StoredRequest | None = None,
        mutation: tuple[str, str] | None = None,
    ) -> JobSnapshot:
        """Publish state, its event, and optional accepted mutation atomically."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "INSERT INTO workflow_events(payload) VALUES ('{}')"
            )
            event_identifier = cursor.lastrowid
            assert event_identifier is not None
            snapshot = snapshot.model_copy(
                update={"latest_event_identifier": event_identifier}
            )
            if request is None:
                connection.execute(
                    "UPDATE workflow_jobs SET payload=? WHERE identifier=?",
                    (snapshot.model_dump_json(), snapshot.job_identifier),
                )
            else:
                connection.execute(
                    "INSERT INTO workflow_jobs VALUES (?,?,?)",
                    (
                        snapshot.job_identifier,
                        snapshot.model_dump_json(),
                        request.model_dump_json(),
                    ),
                )
            event = JobEvent(
                event_identifier=event_identifier,
                job_identifier=snapshot.job_identifier,
                status=snapshot.status,
                phase=snapshot.phase,
                created_at=snapshot.updated_at,
                snapshot=snapshot,
            )
            connection.execute(
                "UPDATE workflow_events SET payload=? WHERE identifier=?",
                (event.model_dump_json(), event_identifier),
            )
            if mutation is not None:
                connection.execute(
                    "INSERT INTO workflow_mutations VALUES (?,?,?)",
                    (*mutation, snapshot.job_identifier),
                )
        return snapshot

    def events(self, after: int) -> list[JobEvent]:
        """Read a bounded ordered replay page for an event cursor."""
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM workflow_events WHERE identifier>? "
                "ORDER BY identifier LIMIT 200",
                (after,),
            ).fetchall()
        return [JobEvent.model_validate_json(row[0]) for row in rows]

    def latest_event(self) -> int:
        """Read the current stream position for snapshot reset responses."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(identifier),0) FROM workflow_events"
            ).fetchone()
        return int(row[0])
