"""Atomic local persistence for configuration, retry results, and job metadata."""

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from backend_service.failures import ApplicationFailure, StorageFailure
from schemas.configuration import ConfigurationProfile
from schemas.jobs import JobSnapshot


class StateStore:
    """Keep validated application state in a versioned SQLite database."""

    def __init__(self, data_directory: Path) -> None:
        """Create the storage location and validate the saved configuration."""
        self.database_path = data_directory / "stegolab.sqlite3"
        try:
            data_directory.mkdir(parents=True, exist_ok=True)
            self._initialize()
            self.get_configuration()
        except (OSError, sqlite3.Error) as failure:
            raise StorageFailure() from failure

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Close each connection and roll back failed transactions."""
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.database_path, timeout=5)
            with connection:
                yield connection
        except (OSError, sqlite3.Error) as failure:
            raise StorageFailure() from failure
        finally:
            if connection is not None:
                connection.close()

    def _initialize(self) -> None:
        """Create version one only for an empty database; reject other versions."""
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 1:
                return
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if version != 0 or tables:
                raise StorageFailure()
            connection.execute(
                "CREATE TABLE configuration "
                "(singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                "payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE mutation_results (request_identifier TEXT PRIMARY KEY, "
                "fingerprint TEXT NOT NULL, response TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE jobs "
                "(job_identifier TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO configuration VALUES (1, ?)",
                (ConfigurationProfile().model_dump_json(),),
            )
            connection.execute("PRAGMA user_version=1")

    @staticmethod
    def _configuration_from_json(payload: str) -> ConfigurationProfile:
        """Validate saved settings instead of quietly replacing damaged values."""
        try:
            stored = json.loads(payload)
            if not isinstance(stored, dict) or set(stored) != set(
                ConfigurationProfile.model_fields
            ):
                raise StorageFailure()
            return ConfigurationProfile.model_validate_json(payload)
        except (ValueError, TypeError) as failure:
            raise StorageFailure() from failure

    def get_configuration(self) -> ConfigurationProfile:
        """Read and validate the single saved configuration."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM configuration WHERE singleton=1"
            ).fetchone()
            if row is None:
                raise StorageFailure()
            return self._configuration_from_json(row[0])

    def update_configuration(
        self,
        request_identifier: str,
        configuration: ConfigurationProfile,
        operation: str = "save",
    ) -> ConfigurationProfile:
        """Save settings and their retry result together in one transaction."""
        validated = ConfigurationProfile.model_validate_json(
            configuration.model_dump_json()
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {"operation": operation, "configuration": validated.model_dump()},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT fingerprint, response FROM mutation_results "
                "WHERE request_identifier=?",
                (request_identifier,),
            ).fetchone()
            if previous is not None:
                if previous[0] != fingerprint:
                    raise ApplicationFailure(
                        "request_identifier_conflict",
                        "This request identifier was already used for a different "
                        "change. Retry with a new request identifier.",
                        409,
                    )
                return self._configuration_from_json(previous[1])
            # Validate before writing so external corruption is never overwritten.
            current = connection.execute(
                "SELECT payload FROM configuration WHERE singleton=1"
            ).fetchone()
            if current is None:
                raise StorageFailure()
            self._configuration_from_json(current[0])
            response = validated.model_dump_json()
            connection.execute(
                "UPDATE configuration SET payload=? WHERE singleton=1", (response,)
            )
            connection.execute(
                "INSERT INTO mutation_results VALUES (?, ?, ?)",
                (request_identifier, fingerprint, response),
            )
        return validated

    @staticmethod
    def _job_from_json(payload: str) -> JobSnapshot:
        """Reject invalid saved job records without returning their contents."""
        try:
            return JobSnapshot.model_validate_json(payload)
        except ValidationError as failure:
            raise StorageFailure() from failure

    def save_job(self, snapshot: JobSnapshot) -> None:
        """Store validated metadata for future services and persistence tests."""
        validated = self._job_from_json(snapshot.model_dump_json())
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?) ON CONFLICT(job_identifier) "
                "DO UPDATE SET payload=excluded.payload",
                (validated.job_identifier, validated.model_dump_json()),
            )

    def list_jobs(self) -> list[JobSnapshot]:
        """Load validated job snapshots in stable creation order."""
        with self._connection() as connection:
            rows = connection.execute("SELECT payload FROM jobs").fetchall()
        snapshots = [self._job_from_json(row[0]) for row in rows]
        return sorted(snapshots, key=lambda job: (job.created_at, job.job_identifier))

    def get_job(self, job_identifier: str) -> JobSnapshot:
        """Read one job or report that no saved job has that identifier."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM jobs WHERE job_identifier=?", (job_identifier,)
            ).fetchone()
        if row is None:
            raise ApplicationFailure(
                "job_not_found", "No saved job was found. Refresh the job list.", 404
            )
        return self._job_from_json(row[0])
