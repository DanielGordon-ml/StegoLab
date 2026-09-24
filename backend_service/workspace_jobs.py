"""One durable local queue supervising isolated CPU workflow processes."""

import fcntl
import logging
import os
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from backend_service.failures import ApplicationFailure, StorageFailure
from backend_service.workflow_preflight import resolve_request, training_preflight
from backend_service.workflow_store import WorkflowStore
from backend_service.workflow_supervision import close_supervisor, run_scheduler
from schemas.configuration import ConfigurationProfile
from schemas.jobs import JobSnapshot
from schemas.workflows import TrainingPreflight, WorkflowActionRequest, WorkflowRequest

if TYPE_CHECKING:
    from backend_service.workspace_catalog import WorkspaceCatalog


class WorkspaceJobService:
    """Own job transitions while existing services own their original proof ledger."""

    def __init__(self, catalog: "WorkspaceCatalog", state_directory: Path) -> None:
        """Prepare durable storage without starting a worker or an experiment."""
        self.catalog = catalog
        self.store = WorkflowStore(state_directory)
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.closing = threading.Event()
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.active_identifier: str | None = None
        self.ownership: int | None = None
        self.logger: logging.Logger | None = None
        self.persistence_failed = False

    def start(self) -> None:
        """Reconcile unfinished metadata and start an empty, single-worker scheduler."""
        with self.lock:
            self.ownership = os.open(
                self.store.path.parent / ".workflow.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
            )
            try:
                fcntl.flock(self.ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(self.ownership)
                self.ownership = None
                raise ApplicationFailure(
                    "workspace_busy", "Another application owns this local queue."
                ) from None
            for job in self.store.list_jobs():
                if job.status in ("queued", "running"):
                    self._change(
                        job,
                        status="interrupted",
                        phase="interrupted",
                        available_actions=[],
                    )
            self.thread = threading.Thread(
                target=run_scheduler, args=(self,), daemon=True
            )
            self.thread.start()

    def close(self) -> None:
        """Stop workers safely even when durable metadata is unavailable."""
        close_supervisor(self)

    def require_storage(self) -> None:
        """Refuse stale responses after losing durable supervisor state."""
        if self.persistence_failed:
            raise StorageFailure()

    def halt_for_storage(self) -> None:
        """Stop admission and request a save independently of failed persistence."""
        self.persistence_failed = True
        self.closing.set()
        self.wake.set()
        if self.logger is not None:
            self.logger.error("workflow_storage_unavailable")

    def list_jobs(self) -> list[JobSnapshot]:
        """Read saved queue and history without touching experiment files."""
        self.require_storage()
        return self.store.list_jobs()

    def get_job(self, identifier: str) -> JobSnapshot:
        """Read one durable snapshot for refresh and reconnection."""
        self.require_storage()
        return self.store.get(identifier)

    def preflight(self, request: WorkflowRequest) -> TrainingPreflight:
        """Inspect fixed training eligibility without spending its allowance."""
        self.require_storage()
        return training_preflight(self.catalog, request)

    def submit(self, request: WorkflowRequest) -> JobSnapshot:
        """Accept one idempotent frozen request after resolving its references."""
        self.require_storage()
        fingerprint = self.store.fingerprint(request.model_dump(mode="json"))
        with self.lock:
            previous = self.store.replay(request.client_request_identifier, fingerprint)
            if previous is not None:
                return previous
            if self.closing.is_set():
                raise ApplicationFailure(
                    "service_stopping",
                    "The application is stopping. Retry after restart.",
                )
            settings: dict[str, object] = request.model_dump(mode="json")
            if request.operation == "train":
                checked = self.preflight(request)
                if not checked.allowed:
                    raise ApplicationFailure(
                        "training_unavailable", " ".join(checked.blockers), 422
                    )
                settings["configuration"] = checked.resolved_settings
                if any(
                    job.operation == "train"
                    and job.experiment_identifier == request.experiment_identifier
                    and job.status in ("queued", "running")
                    for job in self.store.list_jobs()
                ):
                    raise ApplicationFailure(
                        "experiment_busy",
                        "This experiment already has a queued or running job.",
                        409,
                    )
            resolve_request(self.catalog, request)
            now = datetime.now(UTC)
            snapshot = JobSnapshot(
                job_identifier="job_" + uuid4().hex,
                status="queued",
                phase="queued",
                configuration=ConfigurationProfile(),
                available_actions=["cancel"],
                created_at=now,
                updated_at=now,
                operation=request.operation,
                experiment_identifier=request.experiment_identifier,
                frozen_settings=settings,
            )
            accepted = self.store.save(
                snapshot,
                request=request,
                mutation=(request.client_request_identifier, fingerprint),
            )
            self.wake.set()
            return accepted

    def action(self, identifier: str, request: WorkflowActionRequest) -> JobSnapshot:
        """Persist cancel or safe-stop intent before acknowledging a browser retry."""
        self.require_storage()
        fingerprint = self.store.fingerprint(
            {"job": identifier, **request.model_dump(mode="json")}
        )
        with self.lock:
            previous = self.store.replay(request.client_request_identifier, fingerprint)
            if previous is not None:
                return previous
            job = self.store.get(identifier)
            if request.action not in job.available_actions:
                raise ApplicationFailure(
                    "action_unavailable",
                    "This action is no longer available. Refresh the job.",
                    409,
                )
            changes: dict[str, object] = {
                "requested_action": request.action,
                "available_actions": [],
            }
            if request.action == "cancel" and job.status == "queued":
                changes.update(status="cancelled", phase="cancelled")
            elif (
                request.action == "stop"
                and job.status == "running"
                and job.operation == "train"
            ):
                changes["phase"] = "saving"
            else:
                raise ApplicationFailure(
                    "action_unavailable",
                    "This action is not supported for this job.",
                    409,
                )
            return self._change(
                job,
                mutation=(request.client_request_identifier, fingerprint),
                **changes,
            )

    def _change(
        self,
        job: JobSnapshot,
        *,
        mutation: tuple[str, str] | None = None,
        **changes: object,
    ) -> JobSnapshot:
        """Validate and publish a transition together with its ordered event."""
        changes["updated_at"] = datetime.now(UTC)
        updated = JobSnapshot.model_validate(
            job.model_copy(update=changes).model_dump()
        )
        return self.store.save(updated, mutation=mutation)
