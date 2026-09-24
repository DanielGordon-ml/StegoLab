"""Inspect pilot preparation without changing data, spending time, or opening CUDA."""

import platform
import shutil
import time
from pathlib import Path

import torch

from backend_service.failures import ApplicationFailure
from backend_service.pilot_budget import inspect_budget, read_session
from backend_service.pilot_data import load_pilot_data
from schemas.pilot_preflight import (
    PilotPreflightReport,
    PilotPreflightRequest,
    PilotReadinessCheck,
)


def preflight_pilot(request: PilotPreflightRequest) -> PilotPreflightReport:
    """Report metadata, disk, runtime and ledger checks without creating a session."""
    root = Path(request.output_root).resolve()
    existing = root
    while not existing.exists():
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    reserve = max(10 * 1024**3, (usage.total + 9) // 10)
    checks = [
        PilotReadinessCheck(
            name="storage",
            status="passed_locally" if usage.free >= reserve else "failed",
            detail="Keep at least ten GiB or ten percent free after planned writes.",
        )
    ]
    versions_match = (
        platform.python_version_tuple()[:2] == ("3", "12")
        and str(torch.__version__).split("+", 1)[0] == "2.14.0"
        and (request.device == "cpu" or torch.version.cuda == "12.6")
    )
    checks.append(
        PilotReadinessCheck(
            name="runtime_versions",
            status="passed_locally" if versions_match else "failed",
            detail="Use Python 3.12, PyTorch 2.14.0 and the pinned device runtime.",
        )
    )
    report = PilotPreflightReport(
        preparation_passed=False,
        disk_free_bytes=usage.free,
        required_free_reserve_bytes=reserve,
        environment={
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "architecture": platform.machine(),
            "system": platform.system(),
            "compiled_cuda": str(torch.version.cuda),
        },
        checks=[],
    )
    try:
        data = load_pilot_data(Path(request.dataset_directory))
        selection = data.selection
        report.dataset_revision = data.manifest.revision
        report.selection_checksum = selection.selection_checksum
        report.split_counts = dict(data.manifest.split_counts)
        report.training_images = len(data.training_examples)
        report.tuning_images = len(data.tuning_examples)
        report.excluded_tuning_images = len(selection.tuning_excluded)
        checks.extend(
            [
                PilotReadinessCheck(
                    name="dataset_metadata",
                    status="passed_locally",
                    detail="Frozen records and splits verified; image bytes "
                    "are checked on access.",
                ),
                PilotReadinessCheck(
                    name="uhd_profile",
                    status="passed_locally"
                    if selection.gpu_profile_eligible
                    else "failed",
                    detail="The GPU profile requires the frozen full UHD-IQA revision.",
                ),
            ]
        )
    except (ApplicationFailure, OSError, ValueError):
        checks.append(
            PilotReadinessCheck(
                name="dataset_metadata",
                status="failed",
                detail="Prepared dataset metadata is unavailable or inconsistent.",
            )
        )
    ledger = root / "state" / "gpu_pilot"
    try:
        budget = inspect_budget(ledger)
        report.remaining_gpu_seconds = budget.remaining_seconds
        checks.append(
            PilotReadinessCheck(
                name="budget",
                status="passed_locally",
                detail="Read instance-time accounting without starting a session.",
            )
        )
        if request.device == "cuda":
            if (
                request.session_identifier is None
                or budget.active_session_identifier != request.session_identifier
                or budget.remaining_seconds <= 300
            ):
                raise ValueError("A GPU preflight needs an existing host session.")
            session = read_session(ledger, request.session_identifier)
            now = time.time()
            if now < session.last_observed_at or now >= session.checkpoint_at:
                raise ValueError("The selected session is no longer ready for work.")
    except (ApplicationFailure, OSError, ValueError):
        checks.append(
            PilotReadinessCheck(
                name="budget_session",
                status="failed",
                detail="Use an existing GPU session and an intact persistent ledger.",
            )
        )
    available = request.device == "cpu" or torch.cuda.is_available()
    checks.append(
        PilotReadinessCheck(
            name="requested_device",
            status="passed_locally" if available else "failed",
            detail="The requested device is available; execution is checked separately."
            if available
            else "CUDA is unavailable; no CPU fallback was selected.",
        )
    )
    for name in (
        "gpu_training",
        "gpu_resume",
        "gpu_exports",
        "physical_shutdown",
        "data_near_duplicate_audit",
        "release_benchmark",
    ):
        checks.append(
            PilotReadinessCheck(
                name=name,
                status="not_run",
                detail="Requires a later explicit readiness or research gate.",
            )
        )
    report.checks = checks
    report.preparation_passed = all(check.status != "failed" for check in checks)
    return PilotPreflightReport.model_validate(report.model_dump())
