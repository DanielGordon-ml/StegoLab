"""Check learned-package proof accounting without rerunning 74 subprocesses in CI."""

import json
import time
from pathlib import Path
from typing import Literal

import pytest
import torch
from pydantic import ValidationError

from backend_service import model_export_recovery
from backend_service.dataset_reader import DatasetExample
from backend_service.failures import ApplicationFailure
from backend_service.model_data import ProofData
from backend_service.model_evaluation_images import (
    PUBLIC_FIXTURE_PASSWORD,
    PUBLIC_UNICODE_MESSAGE,
)
from backend_service.model_export_recovery import verify_package_recovery
from schemas.datasets import DatasetImageRecord, DatasetManifest
from schemas.export_recovery import ExportRecoveryReport
from schemas.model_exports import ExportFile, ModelExportManifest


def _manifest(role: Literal["encoder", "decoder"]) -> ModelExportManifest:
    """Build valid public metadata while isolating file checks covered elsewhere."""
    record = ExportFile(checksum="a" * 64, size_bytes=1)
    return ModelExportManifest(
        compatibility_identifier="dense_v1_" + "a" * 64,
        source_identifier="public_package_fixture",
        role=role,
        dependencies={"torch": "fixture"},
        producer_environment={"system": "fixture"},
        files={"model.pt2": record, "runtime.py": record},
    )


def _data() -> ProofData:
    """Use partial records only where file reads are explicitly replaced in tests."""
    record = DatasetImageRecord.model_construct()  # type: ignore[call-arg]
    manifest = DatasetManifest.model_construct(revision="b" * 64)  # type: ignore[call-arg]
    crops = tuple(torch.zeros((3, 256, 256)) for _ in range(4))
    return ProofData(
        crops,
        crops,
        manifest,
        {"train": ("a", "b", "c", "d"), "tuning": ("e", "f", "g", "h")},
        tuple(DatasetExample(record, Path("unused")) for _ in range(4)),
    )


def _patch_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace file parsing while keeping scheduling and byte comparison real."""

    def package(directory: Path) -> ModelExportManifest:
        """Resolve one matching fixture role without loading any graph."""
        return _manifest("encoder" if directory.name == "encoder" else "decoder")

    def cover(
        example: DatasetExample,
        destination: Path,
        width: int,
        height: int,
    ) -> None:
        """Write a marker; the mocked runtime checks paths instead of PNG content."""
        destination.write_bytes(b"public cover fixture")

    monkeypatch.setattr(model_export_recovery, "verify_package", package)
    monkeypatch.setattr(model_export_recovery, "_write_cover", cover)


@pytest.mark.parametrize("failure", ["none", "authentication", "extra_newline"])
def test_all_cases_compare_exact_bytes_and_keep_secrets_out_of_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Count every attempt once, detect newline changes, and remove temporary files."""
    _patch_inputs(monkeypatch)
    messages: dict[Path, bytes] = {}
    calls: list[tuple[str, str]] = []
    directories: list[Path] = []
    unicode_seen = False

    def process(
        package: Path,
        arguments: list[str],
        *,
        directory: Path,
        deadline: float | None = None,
        secret_input: bytes | None = None,
    ) -> bytes:
        """Emulate independent runtimes without bypassing the actual byte comparison."""
        nonlocal unicode_seen
        action = arguments[0]
        calls.append((package.name, action))
        directories.append(directory)
        assert PUBLIC_FIXTURE_PASSWORD not in " ".join(arguments)
        if action == "verify":
            assert secret_input is None
            return b""
        assert secret_input is not None and len(secret_input) <= 16_384
        value = json.loads(secret_input)
        assert value["password"] == PUBLIC_FIXTURE_PASSWORD
        image = Path(arguments[arguments.index("--image") + 1])
        assert image.exists()
        if action == "encode":
            assert package.name == "encoder" and set(value) == {"password", "message"}
            output = Path(arguments[arguments.index("--output") + 1])
            assert not output.exists()
            messages[output] = value["message"].encode("utf-8")
            unicode_seen |= value["message"] == PUBLIC_UNICODE_MESSAGE
            output.write_bytes(b"public stego fixture")
            return b""
        assert package.name == "decoder" and set(value) == {"password"}
        result = messages[image]
        if len(messages) == 1 and failure == "authentication":
            raise ApplicationFailure("message_recovery_failed", "Recovery failed.")
        return result + (
            b"\n" if failure == "extra_newline" and len(messages) == 1 else b""
        )

    monkeypatch.setattr(model_export_recovery, "run_export_process", process)
    report = verify_package_recovery(tmp_path, _data())
    expected_trials = 1 if failure == "extra_newline" else 36
    assert report.completed is (failure != "extra_newline")
    assert len(report.trials) == expected_trials
    expected_recovered = {"none": 36, "authentication": 35, "extra_newline": 0}
    assert report.exact_recovery_count == expected_recovered[failure]
    assert report.recovery_gate_passed is (failure == "none")
    assert calls[:2] == [("encoder", "verify"), ("decoder", "verify")]
    assert len(calls) == 2 + 2 * expected_trials
    assert unicode_seen is (failure != "extra_newline")
    if failure == "extra_newline":
        assert report.stopped_reason == "failed"
        assert report.trials[0].error_code == "export_recovery_mismatch"
    assert all(not directory.exists() for directory in directories)
    assert (
        len(
            {
                (trial.source_identity, trial.width, trial.height, trial.fixture_kind)
                for trial in report.trials
            }
        )
        == expected_trials
    )
    serialized = report.model_dump_json()
    assert PUBLIC_FIXTURE_PASSWORD not in serialized
    assert PUBLIC_UNICODE_MESSAGE not in serialized
    assert "MMMM" not in serialized


def test_subprocess_timeout_records_current_attempt_and_stops_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out decoder yields one failed trial and an incomplete overall proof."""
    _patch_inputs(monkeypatch)
    actions: list[str] = []

    def process(
        package: Path,
        arguments: list[str],
        *,
        directory: Path,
        deadline: float | None = None,
        secret_input: bytes | None = None,
    ) -> bytes:
        """Expire the first decoder invocation after two successful preflight checks."""
        actions.append(arguments[0])
        if arguments[0] == "decode":
            raise ApplicationFailure("model_export_budget_expired", "Time expired.")
        return b""

    monkeypatch.setattr(model_export_recovery, "run_export_process", process)
    report = verify_package_recovery(tmp_path, _data())
    assert not report.completed and not report.recovery_gate_passed
    assert report.stopped_reason == "deadline"
    assert len(report.trials) == 1
    assert report.trials[0].error_code == "model_export_budget_expired"
    assert actions == ["verify", "verify", "encode", "decode"]


@pytest.mark.parametrize(
    "field", ["source_identifier", "compatibility_identifier", "role"]
)
def test_mismatched_package_pair_fails_before_any_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Reject two individually valid packages that do not describe the same pair."""

    def package(directory: Path) -> ModelExportManifest:
        """Change only the decoder manifest's selected pair contract field."""
        if directory.name == "encoder":
            return _manifest("encoder")
        changed = {
            "source_identifier": "different_source",
            "compatibility_identifier": "dense_v1_" + "b" * 64,
            "role": "encoder",
        }
        return _manifest("decoder").model_copy(update={field: changed[field]})

    def process(*arguments: object, **keywords: object) -> bytes:
        """Fail if an incompatible pair reaches execution."""
        raise AssertionError("Mismatched package metadata must prevent execution.")

    monkeypatch.setattr(model_export_recovery, "verify_package", package)
    monkeypatch.setattr(model_export_recovery, "run_export_process", process)
    report = verify_package_recovery(tmp_path, _data())
    assert not report.completed and report.stopped_reason == "failed"
    assert report.trials == [] and report.error_code == "model_export_unavailable"


def test_expired_budget_and_inconsistent_report_are_rejected(tmp_path: Path) -> None:
    """An expired proof starts no package reads and cannot claim a false pass."""
    report = verify_package_recovery(tmp_path, _data(), deadline=time.monotonic() - 1)
    assert report.stopped_reason == "deadline" and report.trials == []
    with pytest.raises(ValidationError):
        ExportRecoveryReport.model_validate(
            {**report.model_dump(), "schema_version": True}
        )
    with pytest.raises(ValidationError):
        ExportRecoveryReport.model_validate(
            {**report.model_dump(), "recovery_gate_passed": True}
        )
