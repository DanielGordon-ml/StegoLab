"""Exercise readiness orchestration without accessing CUDA or spending GPU time."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pytest
import torch
from pilot_data_fixtures import pilot_revision
from pydantic import ValidationError

from backend_service import pilot_readiness
from backend_service.failures import ApplicationFailure
from backend_service.pilot_budget import (
    PilotDeadline,
    PilotOperation,
    read_session,
    start_session,
)
from backend_service.pilot_budget_storage import initialize_pilot_budget
from backend_service.pilot_data import load_pilot_data
from backend_service.pilot_readiness_training import _equal
from schemas.pilot_readiness import GpuReadinessRequest


def _request(root: Path, monkeypatch: pytest.MonkeyPatch) -> GpuReadinessRequest:
    """Supply real fixture metadata with a mocked production-profile decision."""
    directory = pilot_revision(root / "dataset")
    data = load_pilot_data(directory)
    data.selection.gpu_profile_eligible = True
    monkeypatch.setattr(pilot_readiness, "load_pilot_data", lambda path: data)
    return GpuReadinessRequest(
        dataset_directory=str(directory),
        output_root=str(root / "output"),
        experiment_identifier="readiness_test",
        session_identifier="existing_session",
    )


def _host_session(
    request: GpuReadinessRequest, stage: Literal["setup", "baseline"] = "setup"
) -> Path:
    """Create only a temporary accounting fixture with no instance or GPU calls."""
    directory = Path(request.output_root) / "state" / "gpu_pilot"
    initialize_pilot_budget(directory)
    start_session(directory, "existing_session", "i-12345678", stage, time.time() - 1)
    return directory


def test_default_report_never_calls_gpu_or_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check-only emits explicit pending checks and cannot acquire GPU resources."""
    request = _request(tmp_path, monkeypatch)

    def forbidden(*arguments: object, **keywords: object) -> None:
        """Reject any hardware or accounting invocation from check-only mode."""
        raise AssertionError("Check-only must not touch CUDA or the host ledger.")

    for name in (
        "PilotOperation",
        "probe_cuda_device",
        "verify_cuda_resume",
        "verify_learned_cuda_exports",
        "verify_learned_cuda_recovery",
    ):
        monkeypatch.setattr(pilot_readiness, name, forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", forbidden)
    report = pilot_readiness.verify_gpu_readiness(request)
    assert report.execution == "check_only" and not report.readiness_passed
    assert all(check.status == "not_run" for check in report.checks[1:])
    assert not (Path(request.output_root) / "state").exists()
    assert (Path(report.report_directory) / "readiness.json").is_file()


def test_run_rejects_missing_or_unregistered_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither an omitted session nor a nonexistent ledger can start GPU work."""
    request = _request(tmp_path, monkeypatch)
    with pytest.raises(ApplicationFailure) as caught:
        pilot_readiness.verify_gpu_readiness(
            request.model_copy(update={"session_identifier": None}), run=True
        )
    assert caught.value.code == "pilot_session"
    with pytest.raises(ApplicationFailure) as caught:
        pilot_readiness.verify_gpu_readiness(request, run=True)
    assert caught.value.code == "pilot_budget_unavailable"
    assert not (Path(request.output_root) / "state").exists()


@pytest.mark.parametrize("learned", [False, True])
def test_explicit_run_holds_one_lease_and_checks_each_profile_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, learned: bool
) -> None:
    """Mock hardware while preserving required order and incomplete-check rules."""
    request = _request(tmp_path, monkeypatch)
    _host_session(request)
    if learned:
        request = request.model_copy(
            update={
                "learned_checkpoint": "frozen_checkpoint",
                "learned_experiment_identifier": "proof_original",
            }
        )
    active = False
    events: list[str] = []

    @contextmanager
    def operation(directory: Path, identifier: str) -> Iterator[PilotDeadline]:
        """Represent exclusive operation ownership without creating a session."""
        nonlocal active
        assert identifier == "existing_session"
        assert directory == Path(request.output_root) / "state" / "gpu_pilot"
        events.append("enter")
        active = True
        yield PilotDeadline(900.0, 800.0, time.monotonic() + 120, time.monotonic() + 90)
        active = False
        events.append("exit")

    def probe() -> dict[str, str]:
        """Record device probing only while the simulated operation lease is held."""
        assert active
        events.append("device")
        return {"device": "mocked_gpu"}

    def resume(
        value: GpuReadinessRequest,
        directory: Path,
        batch: Literal[4, 8],
        deadline: float,
    ) -> list[str]:
        """Verify both batch options use isolated readiness directories."""
        assert active and directory.name == f"batch_{batch}"
        assert value == request and deadline > time.monotonic()
        events.append(f"batch_{batch}")
        return []

    def learned_check(
        value: GpuReadinessRequest, directory: Path, deadline: float
    ) -> list[str]:
        """Require frozen weights for both CUDA export and authentic recovery checks."""
        assert active and value.learned_checkpoint is not None
        events.append(directory.name)
        return []

    monkeypatch.setattr(pilot_readiness, "PilotOperation", operation)
    monkeypatch.setattr(pilot_readiness, "probe_cuda_device", probe)
    monkeypatch.setattr(pilot_readiness, "verify_cuda_resume", resume)
    monkeypatch.setattr(pilot_readiness, "verify_learned_cuda_exports", learned_check)
    monkeypatch.setattr(pilot_readiness, "verify_learned_cuda_recovery", learned_check)
    report = pilot_readiness.verify_gpu_readiness(request, run=True)
    expected = ["enter", "device", "batch_4", "batch_8"]
    if learned:
        expected.extend(["learned_exports", "learned_recovery"])
    assert events == [*expected, "exit"]
    assert report.readiness_passed == learned
    if not learned:
        assert [check.status for check in report.checks[-2:]] == ["not_run", "not_run"]


def test_failed_device_probe_leaves_dependent_checks_unattempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CUDA failure must not become a CPU fallback or a successful readiness run."""
    request = _request(tmp_path, monkeypatch)
    _host_session(request)

    @contextmanager
    def operation(directory: Path, identifier: str) -> Iterator[PilotDeadline]:
        """Hold a fake active session for a failing hardware probe."""
        yield PilotDeadline(900.0, 800.0, time.monotonic() + 120, time.monotonic() + 90)

    def fail() -> dict[str, str]:
        """Return the same explicit device error as an unavailable real GPU."""
        raise ApplicationFailure("pilot_cuda_unavailable", "CUDA is unavailable.")

    monkeypatch.setattr(pilot_readiness, "PilotOperation", operation)
    monkeypatch.setattr(pilot_readiness, "probe_cuda_device", fail)
    report = pilot_readiness.verify_gpu_readiness(request, run=True)
    assert not report.readiness_passed
    assert report.checks[1].status == "failed"
    assert report.checks[1].error_code == "pilot_cuda_unavailable"
    assert all(check.status == "not_run" for check in report.checks[2:])


def test_readiness_rejects_baseline_session_before_any_cuda_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid non-setup lease must not spend the baseline allocation on readiness."""
    request = _request(tmp_path, monkeypatch)
    directory = _host_session(request, "baseline")

    def forbidden(*arguments: object, **keywords: object) -> None:
        """Reject hardware access before checking the held session's stage."""
        raise AssertionError("A baseline session cannot run readiness CUDA checks.")

    monkeypatch.setattr(pilot_readiness, "probe_cuda_device", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", forbidden)
    with pytest.raises(ApplicationFailure) as caught:
        pilot_readiness.verify_gpu_readiness(request, run=True)
    assert caught.value.code == "pilot_readiness_stage"
    session = read_session(directory, "existing_session")
    assert session.stage == "baseline" and session.stopped_at is None
    with PilotOperation(directory, "existing_session"):
        pass


def test_strict_request_and_nested_state_comparison() -> None:
    """Reject loose requests and detect exact changes in safe optimizer trees."""
    base = {"dataset_directory": "dataset", "experiment_identifier": "probe"}
    for change in (
        {"schema_version": 2.0},
        {"learned_checkpoint": "checkpoint"},
        {"unexpected": True},
    ):
        with pytest.raises(ValidationError):
            GpuReadinessRequest.model_validate(base | change)
    first = {"state": [torch.tensor([1.0, 2.0]), (3, "frozen")]}
    assert _equal(first, {"state": [torch.tensor([1.0, 2.0]), (3, "frozen")]})
    assert not _equal(first, {"state": [torch.tensor([1.0, 2.1]), (3, "frozen")]})
