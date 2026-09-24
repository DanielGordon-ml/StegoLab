"""Public research workspace summaries with opaque, server-owned references."""

from typing import Literal

from pydantic import Field

from schemas.base import StrictRecord


class WorkspaceMetrics(StrictRecord):
    """Keep image quality separate from authenticated message recovery."""

    exact_recovery_rate: float | None = Field(default=None, ge=0, le=1)
    trial_count: int | None = Field(default=None, ge=0)
    psnr: float | None = Field(default=None, allow_inf_nan=False)
    psnr_sample_count: int | None = Field(default=None, ge=0)
    ssim: float | None = Field(default=None, ge=-1, le=1)
    ssim_sample_count: int | None = Field(default=None, ge=0)


class WorkspaceDataset(StrictRecord):
    """Describe a prepared revision without exposing its filesystem path."""

    identifier: str
    name: str
    revision: str
    image_count: int = Field(ge=0)
    prepared_bytes: int = Field(ge=0)
    training_images: int = Field(ge=0)
    tuning_images: int = Field(ge=0)
    compatible: bool
    blockers: list[str] = Field(default_factory=list)
    integrity: Literal["not_checked", "verified"] = "not_checked"


class WorkspaceRun(StrictRecord):
    """Associate measured results with one immutable training invocation."""

    identifier: str
    experiment_identifier: str
    status: str
    global_step: int = Field(ge=0)
    dataset_revision: str
    created_at: str
    metrics: WorkspaceMetrics = Field(default_factory=WorkspaceMetrics)


class WorkspaceCheckpoint(StrictRecord):
    """Explain exact-continuation blockers without advertising inference models."""

    identifier: str
    experiment_identifier: str
    dataset_revision: str
    cpu_threads: int | None = Field(default=None, ge=1, le=16)
    global_step: int = Field(ge=0)
    created_at: str
    latest: bool
    pinned: bool
    metrics: WorkspaceMetrics = Field(default_factory=WorkspaceMetrics)
    resume_blockers: list[str] = Field(default_factory=list)


class WorkspaceExport(StrictRecord):
    """Provide a download reference for an independent experimental package."""

    identifier: str
    name: str
    kind: Literal["encoder", "decoder"]
    experimental: Literal[True] = True
    artifact_identifier: str


class WorkspaceSource(StrictRecord):
    """Name a server-approved source root without returning its private path."""

    identifier: str
    label: str


class WorkspaceBudget(StrictRecord):
    """Read the existing shared CPU ledger without changing its accounting."""

    remaining_seconds: float = Field(ge=0, allow_inf_nan=False)
    remaining_experiments: int = Field(ge=0, le=2)
    blocked_reason: str | None = None


class Workspace(StrictRecord):
    """Return discovered local research work and current capability limits."""

    datasets: list[WorkspaceDataset] = Field(default_factory=list)
    runs: list[WorkspaceRun] = Field(default_factory=list)
    checkpoints: list[WorkspaceCheckpoint] = Field(default_factory=list)
    exports: list[WorkspaceExport] = Field(default_factory=list)
    sources: list[WorkspaceSource] = Field(default_factory=list)
    budget: WorkspaceBudget
    warnings: list[str] = Field(default_factory=list)
