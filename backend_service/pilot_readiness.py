"""Explicit later GPU readiness orchestration without launching cloud resources."""

import time
from collections.abc import Callable
from pathlib import Path

from backend_service.failures import ApplicationFailure
from backend_service.pilot_budget import PilotOperation, read_session
from backend_service.pilot_data import load_pilot_data
from backend_service.pilot_readiness_models import (
    verify_learned_cuda_exports,
    verify_learned_cuda_recovery,
)
from backend_service.pilot_readiness_training import (
    probe_cuda_device,
    verify_cuda_resume,
)
from backend_service.pilot_runtime import check_pilot_deadline
from backend_service.proof_runtime import check_output_root, run_directory, write_record
from schemas.pilot_readiness import (
    GpuReadinessCheck,
    GpuReadinessReport,
    GpuReadinessRequest,
)

GPU_CHECKS = (
    "cuda_device",
    "cuda_resume_batch_4",
    "cuda_resume_batch_8",
    "cuda_exports",
    "cuda_png_recovery",
)


def _attempt(
    report: GpuReadinessReport,
    name: str,
    operation: Callable[[], list[str]],
    deadline: float,
) -> bool:
    """Persist a real attempt once without retrying or relaxing its success gate."""
    check = next(check for check in report.checks if check.name == name)
    started = time.monotonic()
    try:
        check_pilot_deadline(deadline)
        check.artifacts = operation()
        check_pilot_deadline(deadline)
        check.status = "passed_on_gpu"
        check.detail = "The required check passed on the selected CUDA device."
    except ApplicationFailure as failure:
        check.status = "failed"
        check.error_code = failure.code
        check.detail = failure.message
    except (OSError, ValueError, RuntimeError, MemoryError, ImportError):
        check.status = "failed"
        check.error_code = "pilot_readiness_failed"
        check.detail = (
            "The GPU check failed. Inspect its saved artifacts and environment."
        )
    check.elapsed_seconds = time.monotonic() - started
    write_record(Path(report.report_directory) / "readiness.json", report)
    return check.status == "passed_on_gpu"


def verify_gpu_readiness(
    request: GpuReadinessRequest, *, run: bool = False
) -> GpuReadinessReport:
    """Inspect metadata by default; require an existing session for every GPU call."""
    request = GpuReadinessRequest.model_validate(request)
    if run and request.session_identifier is None:
        raise ApplicationFailure(
            "pilot_session", "Explicit GPU readiness requires an existing host session."
        )
    root = Path(request.output_root).resolve()
    check_output_root(root, Path(request.dataset_directory).resolve())
    directory = run_directory(root, request.experiment_identifier)
    report = GpuReadinessReport(
        experiment_identifier=request.experiment_identifier,
        execution="explicit_gpu_run" if run else "check_only",
        report_directory=str(directory),
        checks=[
            GpuReadinessCheck(
                name=name,
                status="not_run",
                detail="Requires an explicit GPU run inside an existing host session.",
            )
            for name in GPU_CHECKS
        ],
    )
    try:
        data = load_pilot_data(Path(request.dataset_directory))
        report.dataset_revision = data.manifest.revision
        report.selection_checksum = data.selection.selection_checksum
        if not data.selection.gpu_profile_eligible:
            raise ApplicationFailure(
                "pilot_dataset_ineligible",
                "Use the frozen full UHD-IQA pilot revision.",
            )
        metadata = GpuReadinessCheck(
            name="dataset_metadata",
            status="passed_locally",
            detail="Metadata and frozen selection verified without decoding images.",
        )
    except ApplicationFailure as failure:
        metadata = GpuReadinessCheck(
            name="dataset_metadata",
            status="failed",
            detail=failure.message,
            error_code=failure.code,
        )
    report.checks.insert(0, metadata)
    for check in report.checks:
        if check.name in ("cuda_exports", "cuda_png_recovery") and (
            request.learned_checkpoint is None
        ):
            check.detail = (
                "A frozen learned checkpoint and its original identity are required."
            )
    write_record(directory / "readiness.json", report)
    if not run or metadata.status == "failed":
        return report
    assert request.session_identifier is not None
    with PilotOperation(
        root / "state" / "gpu_pilot", request.session_identifier
    ) as budget:
        session = read_session(root / "state" / "gpu_pilot", request.session_identifier)
        if session.stage != "setup":
            raise ApplicationFailure(
                "pilot_readiness_stage",
                "GPU readiness requires an existing setup session within its "
                "two-hour allocation. Preserve the current session's accounting.",
                422,
            )
        deadline = budget.checkpoint_monotonic_deadline

        def device_probe() -> list[str]:
            """Capture the hardware environment only inside the held operation lease."""
            report.environment = probe_cuda_device()
            return []

        if _attempt(report, "cuda_device", device_probe, deadline):
            _attempt(
                report,
                "cuda_resume_batch_4",
                lambda: verify_cuda_resume(request, directory / "batch_4", 4, deadline),
                deadline,
            )
            _attempt(
                report,
                "cuda_resume_batch_8",
                lambda: verify_cuda_resume(request, directory / "batch_8", 8, deadline),
                deadline,
            )
            if request.learned_checkpoint is not None:
                _attempt(
                    report,
                    "cuda_exports",
                    lambda: verify_learned_cuda_exports(
                        request, directory / "learned_exports", deadline
                    ),
                    deadline,
                )
                _attempt(
                    report,
                    "cuda_png_recovery",
                    lambda: verify_learned_cuda_recovery(
                        request, directory / "learned_recovery", deadline
                    ),
                    deadline,
                )
    report.readiness_passed = all(
        check.status in ("passed_locally", "passed_on_gpu") for check in report.checks
    )
    write_record(directory / "readiness.json", report)
    return report
