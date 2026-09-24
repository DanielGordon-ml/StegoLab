"""Persistent opaque references for approved local workspace artifacts."""

import sqlite3
import uuid
from pathlib import Path

from backend_service.failures import ApplicationFailure, StorageFailure


def safe_path(path: Path, root: Path) -> Path:
    """Reject escapes and links in every component below an approved root."""
    relative = path.absolute().relative_to(root.absolute())
    if any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError("The workspace path is invalid.")
    candidate = root
    if root.is_symlink():
        raise ValueError("Workspace roots cannot be links.")
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Workspace links are unsupported.")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("The workspace path escaped its root.")
    return candidate


class WorkspaceRegistry:
    """Keep path mappings in a separate database without altering source files."""

    def __init__(self, directory: Path, roots: tuple[Path, ...]) -> None:
        """Initialize private identifier storage and approved discovery boundaries."""
        self.roots = roots
        self.database_path = directory / "workspace.sqlite3"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.database_path) as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS workspace_references "
                    "(identifier TEXT PRIMARY KEY, kind TEXT NOT NULL, "
                    "path TEXT NOT NULL, root TEXT NOT NULL, "
                    "UNIQUE(kind, path))"
                )
        except (OSError, sqlite3.Error):
            raise StorageFailure() from None

    def register(self, kind: str, path: Path, root: Path) -> str:
        """Reuse a stable opaque identifier for a real, bounded source path."""
        if root not in self.roots:
            raise ValueError("The workspace root is not registered.")
        safe_path(path, root)
        if not path.exists():
            raise ValueError("The workspace artifact is missing.")
        try:
            with sqlite3.connect(self.database_path, timeout=5) as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO workspace_references VALUES (?, ?, ?, ?)",
                    (
                        uuid.uuid4().hex,
                        kind,
                        str(path.absolute()),
                        str(root.absolute()),
                    ),
                )
                row = connection.execute(
                    "SELECT identifier FROM workspace_references "
                    "WHERE kind=? AND path=?",
                    (kind, str(path.absolute())),
                ).fetchone()
                assert row is not None
                return str(row[0])
        except sqlite3.Error:
            raise StorageFailure() from None

    def resolve(self, identifier: str, kind: str) -> Path:
        """Resolve only a registered reference still inside an approved root."""
        try:
            with sqlite3.connect(self.database_path, timeout=5) as connection:
                row = connection.execute(
                    "SELECT path, root FROM workspace_references "
                    "WHERE identifier=? AND kind=?",
                    (identifier, kind),
                ).fetchone()
            if row is None:
                raise ValueError
            path, root = Path(row[0]), Path(row[1])
            if root not in self.roots:
                raise ValueError
            safe_path(path, root)
            if not path.exists():
                raise ValueError
            return path
        except sqlite3.Error:
            raise StorageFailure() from None
        except (OSError, ValueError):
            raise ApplicationFailure(
                "workspace_reference_missing",
                "This saved item is unavailable. Refresh the workspace and select "
                "an available item.",
                404,
            ) from None
