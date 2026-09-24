"""Check pilot preparation boundaries and independent runtime device contracts."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from pilot_data_fixtures import pilot_revision
from pydantic import ValidationError

from backend_service.command_line import main
from backend_service.failures import ApplicationFailure
from backend_service.model_export_io import verify_package
from backend_service.model_export_runtime import load_export
from backend_service.model_export_verification import run_export_process
from backend_service.model_exports import _package
from backend_service.model_networks import build_models, model_pair_identifier
from backend_service.pilot_budget import start_session
from backend_service.pilot_budget_storage import initialize_pilot_budget
from backend_service.pilot_data import load_pilot_data
from backend_service.pilot_export import cuda_runtime_dependencies
from backend_service.pilot_preflight import preflight_pilot
from schemas.model_exports import ExportMetadata, ModelExportManifest
from schemas.pilot_preflight import PilotPreflightRequest


def test_preflight_reads_without_creating_state_or_claiming_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local preparation reports pending hardware checks and leaves state untouched."""
    data = load_pilot_data(pilot_revision(tmp_path / "fixtures"))
    data.selection.gpu_profile_eligible = True
    monkeypatch.setattr(
        "backend_service.pilot_preflight.load_pilot_data", lambda _: data
    )
    output = tmp_path / "not_created"
    result = preflight_pilot(
        PilotPreflightRequest(
            dataset_directory="fixture",
            output_root=str(output),
        )
    )
    assert result.preparation_passed and not output.exists()
    assert result.remaining_gpu_seconds == 86400
    assert not result.pilot_ready and not result.gpu_checks_run
    statuses = {check.name: check.status for check in result.checks}
    assert statuses["gpu_resume"] == statuses["gpu_exports"] == "not_run"
    assert result.training_images == 5 and result.tuning_images == 2


def test_preflight_exposes_failed_data_storage_and_cuda_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent blockers remain visible without silently selecting CPU."""
    usage = shutil.disk_usage(tmp_path)._replace(free=1)
    monkeypatch.setattr(
        "backend_service.pilot_preflight.shutil.disk_usage", lambda _: usage
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = preflight_pilot(
        PilotPreflightRequest(
            dataset_directory=str(tmp_path / "missing"),
            output_root=str(tmp_path),
            device="cuda",
        )
    )
    assert not result.preparation_passed
    failed = {check.name for check in result.checks if check.status == "failed"}
    assert {
        "storage",
        "dataset_metadata",
        "requested_device",
        "budget_session",
    } <= failed


def test_cli_routes_version_two_and_returns_failed_preflight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public command uses strict records and a nonzero failed-check status."""
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_directory": str(tmp_path / "missing"),
                "output_root": str(tmp_path),
            }
        )
    )
    assert main(["preflight_pilot", str(request)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == 2 and not result["preparation_passed"]
    assert result["gpu_checks_run"] is False
    request.write_text(json.dumps({"schema_version": 2, "secret": "do_not_echo"}))
    assert main(["train", str(request)]) == 2
    assert "do_not_echo" not in capsys.readouterr().err


def test_version_two_package_keeps_separate_runtime_pins_and_cpu_execution(
    tmp_path: Path,
) -> None:
    """A new isolated package executes on CPU and carries frozen CUDA requirements."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        encoder, decoder = build_models(17)
        package = tmp_path / "decoder"
        _package(
            decoder,
            package,
            "decoder",
            ExportMetadata(
                compatibility_identifier=model_pair_identifier(encoder, decoder),
                source_identifier="public_pilot_package_fixture",
            ),
            None,
            {"cuda": cuda_runtime_dependencies()},
        )
        manifest = verify_package(package)
        assert manifest.format_version == 2
        assert manifest.runtime_dependencies["cpu"] == manifest.dependencies
        assert manifest.runtime_dependencies["cuda"]["torch"] == "2.14.0+cu126"
        assert "nvidia-cudnn-cu12" in manifest.runtime_dependencies["cuda"]
        assert (package / "requirements-cpu.txt").is_file()
        assert (package / "requirements-cuda.txt").is_file()
        assert not any("training" in name for name in manifest.files)
        run_export_process(package, ["verify", "--device", "cpu"], directory=tmp_path)
    finally:
        torch.set_num_threads(threads)


def test_cuda_runtime_rejection_happens_before_package_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable hardware must not trigger a hidden CPU package execution."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ApplicationFailure) as caught:
        load_export(tmp_path / "does_not_exist", device="cuda")
    assert caught.value.code == "cuda_unavailable"


@pytest.mark.parametrize("value", [True, 1.0, 2.0])
def test_package_format_version_rejects_lookalikes(value: object) -> None:
    """Reject alternate numeric types before reading graph or dependency records."""
    with pytest.raises(ValidationError) as caught:
        ModelExportManifest.model_validate({"format_version": value})
    assert any(error["loc"] == ("format_version",) for error in caught.value.errors())


def test_preflight_version_requires_an_integer() -> None:
    """Apply the same strict version boundary to read-only request records."""
    with pytest.raises(ValidationError):
        PilotPreflightRequest.model_validate(
            {
                "schema_version": 2.0,
                "dataset_directory": "unused",
            }
        )


def test_gpu_preflight_rejects_expired_stage_without_writing_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remaining project hours cannot extend an expired setup-session deadline."""
    data = load_pilot_data(pilot_revision(tmp_path / "fixtures"))
    data.selection.gpu_profile_eligible = True
    monkeypatch.setattr(
        "backend_service.pilot_preflight.load_pilot_data", lambda _: data
    )
    monkeypatch.setattr("time.time", lambda: 1000.0)
    ledger = tmp_path / "state" / "gpu_pilot"
    initialize_pilot_budget(ledger)
    start_session(
        ledger, "setup", "i-12345678", "setup", 1000.0, requested_seconds=600.0
    )
    before = (ledger / "ledger.json").read_bytes()
    monkeypatch.setattr("time.time", lambda: 1700.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    report = preflight_pilot(
        PilotPreflightRequest(
            dataset_directory="fixture",
            output_root=str(tmp_path),
            device="cuda",
            session_identifier="setup",
        )
    )
    assert not report.preparation_passed
    assert report.remaining_gpu_seconds == 85700.0
    assert any(
        check.name == "budget_session" and check.status == "failed"
        for check in report.checks
    )
    assert (ledger / "ledger.json").read_bytes() == before
