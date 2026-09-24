"""Explicit installation of verified experimental model pairs for inference."""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from backend_service.failures import ApplicationFailure, StorageFailure
from backend_service.model_export_io import read_export_file, verify_package
from backend_service.workflow_store import WorkflowStore
from backend_service.workspace_registry import safe_path
from schemas.capabilities import Capabilities
from schemas.jobs import JobSnapshot
from schemas.model_exports import ExportVerification, ModelExportManifest
from schemas.models import InstalledModel, ModelInstallRequest


def pair_failure(code: str, message: str) -> ApplicationFailure:
    """Describe an unusable export pair without exposing its location."""
    return ApplicationFailure(code, message, 422)


def verify_model_pair(
    pair_root: Path,
) -> tuple[ModelExportManifest, ExportVerification]:
    """Check both packages and the recorded pair verification without loading graphs."""
    roles = (pair_root / "encoder", pair_root / "decoder")
    if pair_root.is_symlink() or any(
        role.is_symlink() or not role.is_dir() for role in roles
    ):
        raise pair_failure(
            "model_pair_incomplete",
            "Install needs an encoder and a decoder package exported together.",
        )
    encoder = verify_package(pair_root / "encoder")
    decoder = verify_package(pair_root / "decoder")
    if (
        encoder.role != "encoder"
        or decoder.role != "decoder"
        or encoder.compatibility_identifier != decoder.compatibility_identifier
        or encoder.source_identifier != decoder.source_identifier
        or encoder.format_version != decoder.format_version
    ):
        raise pair_failure(
            "model_pair_incomplete",
            "Install needs an encoder and a decoder package exported together.",
        )
    record = pair_root / "verification.json"
    try:
        if record.is_symlink() or not record.is_file():
            raise ValueError
        verification = ExportVerification.model_validate_json(
            read_export_file(record, 64 * 1024)
        )
        if verification.compatibility_identifier != encoder.compatibility_identifier:
            raise ValueError
    except (OSError, ValueError, ValidationError, ApplicationFailure):
        raise pair_failure(
            "model_verification_missing",
            "This export has no matching verification record. Export the "
            "checkpoint again before installing it.",
        ) from None
    return encoder, verification


def model_in_use(jobs: list[JobSnapshot], model_identifier: str) -> bool:
    """Report whether a queued or running inference job still references a model."""
    return any(
        job.operation in ("encode", "decode")
        and job.status in ("queued", "running")
        and job.frozen_settings.get("model_identifier") == model_identifier
        for job in jobs
    )


class InstalledModelStore:
    """Keep explicit installation records separately from export directories."""

    def __init__(self, directory: Path, roots: tuple[Path, ...]) -> None:
        """Create the private installation database inside the application state."""
        self.roots = roots
        self.path = directory / "installed_models.sqlite3"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with self.connection() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS installed_models "
                    "(model_identifier TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                    "package_directory TEXT NOT NULL, root TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS install_mutations "
                    "(identifier TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
                    "model_identifier TEXT NOT NULL)"
                )
        except OSError:
            raise StorageFailure() from None

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

    def _available_directory(self, directory: Path, root: Path) -> Path:
        """Re-check a recorded package directory against its approved root."""
        if root not in self.roots:
            raise ValueError("The recorded root is not approved.")
        candidate = safe_path(directory, root)
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError("The recorded package directory is missing.")
        return candidate

    def install(
        self, request: ModelInstallRequest, locate: Callable[[], tuple[Path, Path]]
    ) -> tuple[InstalledModel, bool]:
        """Record one verified pair, replaying identical retries safely.

        The pair location is resolved only after the retry check so that a reused
        identifier with different content is refused before any lookup.
        """
        fingerprint = WorkflowStore.fingerprint(
            {"operation": "install", "export_reference": request.export_reference}
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            replay = connection.execute(
                "SELECT fingerprint, model_identifier FROM install_mutations "
                "WHERE identifier=?",
                (request.client_request_identifier,),
            ).fetchone()
            if replay is not None:
                if replay[0] != fingerprint:
                    raise ApplicationFailure(
                        "request_identifier_conflict",
                        "This request identifier was used for another change. "
                        "Use a new identifier for a new installation.",
                        409,
                    )
                return self._read(connection, str(replay[1])), False
            pair_root, root = locate()
            directory = self._available_directory(pair_root, root)
            manifest, _ = verify_model_pair(directory)
            model, created = self._record(
                connection, request, directory, root, manifest
            )
            connection.execute(
                "INSERT INTO install_mutations VALUES (?,?,?)",
                (
                    request.client_request_identifier,
                    fingerprint,
                    model.model_identifier,
                ),
            )
            return model, created

    def _record(
        self,
        connection: sqlite3.Connection,
        request: ModelInstallRequest,
        directory: Path,
        root: Path,
        manifest: ModelExportManifest,
    ) -> tuple[InstalledModel, bool]:
        """Insert a new installation or confirm an identical existing one."""
        existing = connection.execute(
            "SELECT payload FROM installed_models WHERE model_identifier=?",
            (directory.name,),
        ).fetchone()
        if existing is not None:
            model = InstalledModel.model_validate_json(existing[0])
            if model.compatibility_identifier != manifest.compatibility_identifier:
                raise ApplicationFailure(
                    "model_identifier_conflict",
                    "A different model pair is installed under this name. Remove "
                    "it before installing another pair with the same name.",
                    409,
                )
            return model, False
        try:
            model = InstalledModel(
                model_identifier=directory.name,
                compatibility_identifier=manifest.compatibility_identifier,
                source_identifier=manifest.source_identifier,
                format_version=manifest.format_version,
                installed_at=datetime.now(UTC),
                export_reference=request.export_reference,
            )
        except ValidationError:
            raise pair_failure(
                "model_identifier_invalid",
                "Use letters, digits, dots, dashes, and underscores for the export "
                "folder name, up to 64 characters.",
            ) from None
        connection.execute(
            "INSERT INTO installed_models VALUES (?,?,?,?)",
            (
                model.model_identifier,
                model.model_dump_json(),
                str(directory.absolute()),
                str(root.absolute()),
            ),
        )
        return model, True

    def _read(
        self, connection: sqlite3.Connection, model_identifier: str
    ) -> InstalledModel:
        """Return one recorded installation whose package is still available."""
        row = connection.execute(
            "SELECT payload, package_directory, root FROM installed_models "
            "WHERE model_identifier=?",
            (model_identifier,),
        ).fetchone()
        try:
            if row is None:
                raise ValueError
            self._available_directory(Path(row[1]), Path(row[2]))
            return InstalledModel.model_validate_json(row[0])
        except (OSError, ValueError, ValidationError):
            raise ApplicationFailure(
                "model_not_installed",
                "This model is not installed or its package is missing. Install "
                "an exported model pair first.",
                404,
            ) from None

    def list_models(self) -> list[InstalledModel]:
        """List installed models whose package directories are still present."""
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT model_identifier FROM installed_models "
                "ORDER BY model_identifier"
            ).fetchall()
            result: list[InstalledModel] = []
            for row in rows:
                try:
                    result.append(self._read(connection, str(row[0])))
                except ApplicationFailure:
                    continue
            return result

    def resolve(self, model_identifier: str) -> tuple[InstalledModel, Path]:
        """Return one installed model and its verified package directory."""
        with self.connection() as connection:
            model = self._read(connection, model_identifier)
            row = connection.execute(
                "SELECT package_directory, root FROM installed_models "
                "WHERE model_identifier=?",
                (model_identifier,),
            ).fetchone()
            assert row is not None
            return model, self._available_directory(Path(row[0]), Path(row[1]))

    def remove(self, model_identifier: str) -> None:
        """Forget an installation without touching the export directory."""
        with self.connection() as connection:
            cursor = connection.execute(
                "DELETE FROM installed_models WHERE model_identifier=?",
                (model_identifier,),
            )
            if cursor.rowcount == 0:
                raise ApplicationFailure(
                    "model_not_installed",
                    "This model is not installed. Refresh the model list.",
                    404,
                )

    def capabilities(self) -> Capabilities:
        """Derive advertised features only from currently installed models."""
        models = self.list_models()
        if not models:
            return Capabilities()
        return Capabilities(
            available_models=[model.model_identifier for model in models],
            available_profiles=["test_only_v1"],
            encoding_available=True,
            decoding_available=True,
            maximum_payload_bytes=max(model.maximum_payload_bytes for model in models),
            minimum_image_side=min(model.minimum_side for model in models),
            maximum_image_side=max(model.maximum_side for model in models),
        )
