"""Read-only workspace discovery backed by private persistent opaque references."""

import json
import os
from pathlib import Path

from backend_service.failures import ApplicationFailure
from backend_service.workspace_artifacts import export_archive
from backend_service.workspace_eligibility import (
    checkpoint_eligibility,
    dataset_eligibility,
    read_budget,
)
from backend_service.workspace_registry import WorkspaceRegistry, safe_path
from backend_service.workspace_reports import discover
from backend_service.workspace_views import (
    discover_checkpoints,
    discover_exports,
    discover_runs,
)
from schemas.workspace import Workspace, WorkspaceDataset, WorkspaceSource

DATA_ROOT_LABEL = "Downloaded and uploaded sources"
MAXIMUM_SOURCE_FOLDERS = 200


def with_data_root(roots: dict[str, Path], root: Path) -> dict[str, Path]:
    """Offer the workspace data folder as a source unless it is already listed."""
    directory = root / "data"
    if not directory.is_dir() or any(
        path.absolute() == directory.absolute() for path in roots.values()
    ):
        return roots
    return {
        **roots,
        ("Local source images" if not roots else DATA_ROOT_LABEL): directory,
    }


def source_folders(path: Path) -> list[str]:
    """List visible real subfolders of a source root, sorted and bounded."""
    names: list[str] = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False) and not entry.name.startswith(
                    "."
                ):
                    names.append(entry.name)
    except OSError:
        return []
    return sorted(names)[:MAXIMUM_SOURCE_FOLDERS]


class WorkspaceCatalog:
    """Discover existing research without resetting budgets or rewriting artifacts."""

    def __init__(
        self,
        root: Path,
        state_directory: Path,
        *,
        source_roots: dict[str, Path] | None = None,
    ) -> None:
        """Use one original output root and explicitly approved source folders."""
        self.root = root.absolute()
        self.source_roots = (
            source_roots if source_roots is not None else self._source_roots()
        )
        self.source_roots = {
            name: path.absolute() for name, path in self.source_roots.items()
        }
        self.registry = WorkspaceRegistry(
            state_directory, (self.root, *self.source_roots.values())
        )

    def _source_roots(self) -> dict[str, Path]:
        """Read trusted host configuration, keeping paths out of browser payloads."""
        value = os.environ.get("STEGOLAB_SOURCE_ROOTS")
        if value is None:
            return with_data_root({}, self.root)
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, dict) or len(parsed) > 20:
                raise ValueError
            if any(
                not isinstance(label, str)
                or not label
                or len(label) > 120
                or not isinstance(path, str)
                or not Path(path).is_absolute()
                for label, path in parsed.items()
            ):
                raise ValueError
            roots = {label: Path(path) for label, path in parsed.items()}
        except ValueError:
            raise ApplicationFailure(
                "workspace_source_configuration",
                "The server source-folder configuration is invalid. "
                "Ask the server owner to check the registered folders.",
            ) from None
        return with_data_root(roots, self.root)

    def _datasets(self) -> tuple[list[WorkspaceDataset], list[str]]:
        """Inspect frozen metadata and clearly defer full image-byte verification."""
        datasets: list[WorkspaceDataset] = []
        warnings: list[str] = []
        paths = discover(
            self.root,
            ("datasets/*/*/manifest.json", ".runtime/*/datasets/*/*/manifest.json"),
        )
        for path in paths:
            try:
                manifest, blockers = dataset_eligibility(path.parent)
                datasets.append(
                    WorkspaceDataset(
                        identifier=self.registry.register(
                            "dataset", path.parent, self.root
                        ),
                        name=manifest.selection_name or manifest.dataset_name,
                        revision=manifest.revision,
                        image_count=manifest.accepted_count,
                        prepared_bytes=manifest.prepared_bytes,
                        training_images=manifest.unique_eligible_by_split.get(
                            "train", 0
                        ),
                        tuning_images=manifest.unique_eligible_by_split.get(
                            "tuning", 0
                        ),
                        compatible=not blockers,
                        blockers=blockers,
                    )
                )
            except (OSError, ValueError, ApplicationFailure):
                warnings.append("A prepared dataset could not be read or verified.")
        return datasets, warnings

    def snapshot(self) -> Workspace:
        """Refresh stable references and summaries without touching source evidence."""
        datasets, warnings = self._datasets()
        identities = {item.revision: item.blockers for item in datasets}
        sources: list[WorkspaceSource] = []
        for label, path in self.source_roots.items():
            try:
                sources.append(
                    WorkspaceSource(
                        identifier=self.registry.register("source", path, path),
                        label=label,
                        folders=source_folders(path),
                    )
                )
            except (OSError, ValueError, ApplicationFailure):
                warnings.append("A registered source folder is unavailable.")
        return Workspace(
            datasets=datasets,
            runs=discover_runs(self.root, self.registry),
            checkpoints=discover_checkpoints(self.root, self.registry, identities),
            exports=discover_exports(self.root, self.registry),
            sources=sources,
            budget=read_budget(self.root),
            warnings=sorted(set(warnings)),
        )

    def resolve_dataset(self, identifier: str) -> Path:
        """Return one registered prepared revision for a server-owned operation."""
        return self.registry.resolve(identifier, "dataset")

    def resolve_checkpoint(self, identifier: str) -> Path:
        """Return one registered checkpoint for a server-owned operation."""
        return self.registry.resolve(identifier, "checkpoint")

    def register_checkpoint(self, path: Path) -> str:
        """Register a newly verified worker result under the original workspace."""
        safe_path(path, self.root)
        from backend_service.training_checkpoints import inspect_checkpoint

        inspect_checkpoint(path)
        return self.registry.register("checkpoint", path, self.root)

    def resolve_source(self, identifier: str, source_subdirectory: str = "") -> Path:
        """Resolve a relative source folder inside its explicitly registered root."""
        root = self.registry.resolve(identifier, "source")
        try:
            relative = Path(source_subdirectory)
            if relative.is_absolute() or "\\" in source_subdirectory:
                raise ValueError
            path = safe_path(root / relative, root)
            if not path.is_dir():
                raise ValueError
            return path
        except (OSError, ValueError):
            raise ApplicationFailure(
                "workspace_source_invalid",
                "Choose an existing folder inside the selected source location.",
                422,
            ) from None

    def dataset_blockers(self, identifier: str) -> list[str]:
        """Return current metadata eligibility before a bounded worker validation."""
        try:
            return dataset_eligibility(self.resolve_dataset(identifier))[1]
        except (OSError, ValueError):
            raise ApplicationFailure(
                "workspace_dataset_invalid",
                "The selected dataset changed. Validate it before starting work.",
                422,
            ) from None

    def checkpoint_blockers(self, identifier: str) -> list[str]:
        """Check exact continuation against the current code, data and environment."""
        datasets, _ = self._datasets()
        identities = {item.revision: item.blockers for item in datasets}
        return checkpoint_eligibility(
            self.resolve_checkpoint(identifier), identities, self.root
        )[1]

    def download_export(self, identifier: str) -> tuple[bytes, str]:
        """Download only a registered, checksummed independent model package."""
        return export_archive(self.registry.resolve(identifier, "export"))
