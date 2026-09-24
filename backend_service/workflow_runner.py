"""Execute one queued job and publish verified final metadata."""

import subprocess
from typing import TYPE_CHECKING

from backend_service.failures import ApplicationFailure
from backend_service.workflow_execution import execute_command, public_result
from backend_service.workflow_preflight import resolve_request
from schemas.training import TrainingStep

if TYPE_CHECKING:
    from backend_service.workspace_jobs import WorkspaceJobService


def run_job(self: "WorkspaceJobService", identifier: str) -> None:
    """Resolve frozen references, run a child, and verify its final publication."""
    request = self.store.request(identifier)
    if request.operation == "train":
        checked = self.preflight(request)
        if not checked.allowed:
            raise ApplicationFailure(
                "training_unavailable", " ".join(checked.blockers), 422
            )
    command, document = resolve_request(self.catalog, request)

    def remember_process(process: subprocess.Popen[bytes] | None) -> None:
        """Track process ownership only while this invocation is active."""
        self.process = process

    def stop_requested() -> bool:
        """Read persisted stop intent, including application shutdown."""
        return (
            self.closing.is_set()
            or self.store.get(identifier).requested_action == "stop"
        )

    def progress(step: TrainingStep) -> None:
        """Publish measured scalar progress without inventing an ETA."""
        with self.lock:
            job = self.store.get(identifier)
            self._change(
                job,
                phase="saving" if job.requested_action else "training",
                progress=min(1.0, step.global_step / request.stop_after_step),
                metrics=step.model_dump(mode="json"),
            )

    code, result = execute_command(
        self.catalog.root,
        command,
        document,
        on_process=remember_process,
        on_progress=progress,
        stop_requested=stop_requested,
    )
    published = public_result(result)
    checkpoint = result.get("checkpoint")
    if isinstance(checkpoint, str):
        from backend_service.training_checkpoints import inspect_checkpoint

        path = self.catalog.root / checkpoint
        if not path.resolve().is_relative_to(
            (self.catalog.root / "checkpoints").resolve()
        ):
            raise ValueError("Checkpoint escaped the registered workspace.")
        inspect_checkpoint(path)
        published["checkpoint_identifier"] = self.catalog.register_checkpoint(path)
    success = code == 0 or (
        request.operation == "train" and code in (130, 143) and checkpoint is not None
    )
    status = "completed" if success else "failed"
    if (
        request.operation == "train"
        and result.get("status") in ("stopped", "budget_exhausted")
        and success
    ):
        status = "stopped"
    with self.lock:
        self._change(
            self.store.get(identifier),
            status=status,
            phase=status,
            available_actions=[],
            result=published,
            progress=1.0
            if status == "completed"
            else self.store.get(identifier).progress,
            error=None
            if success
            else {
                "code": "workflow_incomplete",
                "message": "The operation ended before all checks finished. "
                "Review its saved results.",
            },
        )
