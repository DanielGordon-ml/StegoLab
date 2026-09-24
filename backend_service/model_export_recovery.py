"""Bounded exact-message proof using only independent package subprocesses."""

import json
import tempfile
import time
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from backend_service.dataset_reader import DatasetExample
from backend_service.failures import ApplicationFailure
from backend_service.model_data import ProofData, load_center_crop
from backend_service.model_evaluation_images import (
    PUBLIC_FIXTURE_PASSWORD,
    PUBLIC_UNICODE_MESSAGE,
    FixtureKind,
)
from backend_service.model_export_io import export_failure, verify_package
from backend_service.model_export_verification import run_export_process
from backend_service.model_exports import check_export_deadline
from backend_service.payload_capacity import calculate_capacity
from schemas.export_recovery import ExportRecoveryReport, ExportRecoveryTrial
from schemas.protocol import ProtocolContext

RECOVERY_SIZES = ((512, 512), (513, 517), (1024, 1024))
RECOVERY_KINDS: tuple[FixtureKind, ...] = ("empty", "unicode", "maximum")


def _secret_request(message: str | None = None) -> bytes:
    """Serialize bounded public test inputs for protected stdin, never arguments."""
    value = {"password": PUBLIC_FIXTURE_PASSWORD}
    if message is not None:
        value["message"] = message
    content = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    if len(content) > 16_384:
        raise export_failure()
    return content


def _message(kind: FixtureKind, context: ProtocolContext) -> str:
    """Use the same public exact-byte fixtures as the eager-model proof."""
    if kind == "empty":
        return ""
    if kind == "unicode":
        return PUBLIC_UNICODE_MESSAGE
    return "M" * calculate_capacity(context).maximum_message_bytes


def _write_cover(
    example: DatasetExample, destination: Path, width: int, height: int
) -> None:
    """Write an explicit center crop with original RGB integer values."""
    cover = load_center_crop(example, width, height)
    pixels = torch.round(cover * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
    with Image.fromarray(pixels) as image:
        image.save(destination, format="PNG")


def _trial(
    package_root: Path,
    example: DatasetExample,
    source_identity: str,
    compatibility_identifier: str,
    width: int,
    height: int,
    kind: FixtureKind,
    scratch: Path,
    deadline: float | None,
) -> ExportRecoveryTrial:
    """Encode and decode once in separate processes, preserving every output byte."""
    started = time.monotonic()
    recovered = False
    error_code: str | None = None
    try:
        check_export_deadline(deadline)
        context = ProtocolContext(
            width=width,
            height=height,
            compatibility_identifier=compatibility_identifier,
        )
        message = _message(kind, context)
        with tempfile.TemporaryDirectory(prefix="trial-", dir=scratch) as temporary:
            directory = Path(temporary)
            cover, stego = directory / "cover.png", directory / "stego.png"
            _write_cover(example, cover, width, height)
            check_export_deadline(deadline)
            run_export_process(
                package_root / "encoder",
                ["encode", "--image", str(cover), "--output", str(stego)],
                directory=directory,
                deadline=deadline,
                secret_input=_secret_request(message),
            )
            restored = run_export_process(
                package_root / "decoder",
                ["decode", "--image", str(stego)],
                directory=directory,
                deadline=deadline,
                secret_input=_secret_request(),
            )
            recovered = restored == message.encode("utf-8", errors="strict")
            if not recovered:
                error_code = "export_recovery_mismatch"
    except ApplicationFailure as failure:
        error_code = failure.code
    except (OSError, ValueError, RuntimeError, MemoryError):
        error_code = "export_recovery_failed"
    return ExportRecoveryTrial(
        source_identity=source_identity,
        width=width,
        height=height,
        fixture_kind=kind,
        recovered=recovered,
        elapsed_seconds=time.monotonic() - started,
        error_code=error_code,
    )


def verify_package_recovery(
    package_root: Path,
    data: ProofData,
    *,
    deadline: float | None = None,
) -> ExportRecoveryReport:
    """Record 36 trials or an explicit partial result without loading graphs here."""
    started = time.monotonic()
    compatibility: str | None = None
    source: str | None = None
    trials: list[ExportRecoveryTrial] = []
    reason: Literal["complete", "deadline", "failed"] = "complete"
    error_code: str | None = None
    try:
        check_export_deadline(deadline)
        encoder = verify_package(package_root / "encoder")
        check_export_deadline(deadline)
        decoder = verify_package(package_root / "decoder")
        if (
            encoder.role != "encoder"
            or decoder.role != "decoder"
            or encoder.compatibility_identifier != decoder.compatibility_identifier
            or encoder.source_identifier != decoder.source_identifier
            or len(data.tuning_examples) != 4
            or len(data.identities.get("tuning", ())) != 4
        ):
            raise export_failure()
        compatibility, source = (
            encoder.compatibility_identifier,
            encoder.source_identifier,
        )
        with tempfile.TemporaryDirectory(prefix="stegolab-package-recovery-") as name:
            scratch = Path(name)
            for role in ("encoder", "decoder"):
                run_export_process(
                    package_root / role,
                    ["verify"],
                    directory=scratch,
                    deadline=deadline,
                )
            for index, example in enumerate(data.tuning_examples):
                for width, height in RECOVERY_SIZES:
                    for kind in RECOVERY_KINDS:
                        check_export_deadline(deadline)
                        trial = _trial(
                            package_root,
                            example,
                            data.identities["tuning"][index],
                            compatibility,
                            width,
                            height,
                            kind,
                            scratch,
                            deadline,
                        )
                        trials.append(trial)
                        if trial.error_code == "model_export_budget_expired":
                            raise ApplicationFailure(
                                trial.error_code, "The package proof time limit ended."
                            )
                        if trial.error_code not in (None, "message_recovery_failed"):
                            assert trial.error_code is not None
                            raise ApplicationFailure(
                                trial.error_code,
                                "The independent package recovery check failed.",
                            )
    except ApplicationFailure as failure:
        error_code = failure.code
        reason = "deadline" if error_code == "model_export_budget_expired" else "failed"
    except (OSError, ValueError, RuntimeError, MemoryError):
        error_code, reason = "export_recovery_failed", "failed"
    recovered = sum(trial.recovered for trial in trials)
    complete = reason == "complete" and len(trials) == 36
    return ExportRecoveryReport(
        compatibility_identifier=compatibility,
        source_identifier=source,
        dataset_revision=data.manifest.revision,
        trials=trials,
        completed=complete,
        exact_recovery_count=recovered,
        exact_recovery_rate=recovered / len(trials) if trials else None,
        recovery_gate_passed=complete and recovered == 36,
        stopped_reason=reason,
        elapsed_seconds=time.monotonic() - started,
        error_code=error_code,
    )
