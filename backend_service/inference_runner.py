"""Run one encode or decode job in isolated package processes without saving secrets."""

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from backend_service.failures import ApplicationFailure
from backend_service.inference_execution import run_package, secret_input
from backend_service.inference_jobs import INPUTS_LOST, require_inference
from backend_service.model_export_io import export_failure
from backend_service.model_installation import verify_model_pair
from schemas.inference_jobs import (
    MAXIMUM_MESSAGE_BYTES,
    DecodingJobResult,
    EncodingJobResult,
    InferenceJobRecord,
)

if TYPE_CHECKING:
    from backend_service.inference_jobs import InferenceServices
    from backend_service.workspace_jobs import WorkspaceJobService

PhaseReporter = Callable[[str], None]
ProcessObserver = Callable[[subprocess.Popen[bytes] | None], None]


def verification_failure() -> ApplicationFailure:
    """Explain that nothing was published because the decoder could not read it."""
    return ApplicationFailure(
        "encoding_verification_failed",
        "The saved image did not pass message recovery with the matching decoder. "
        "Nothing was published. Try another cover image.",
        422,
    )


def encode_image(
    inference: "InferenceServices",
    package_root: Path,
    image: Path,
    scratch: Path,
    secrets: dict[str, str],
    record: InferenceJobRecord,
    on_process: ProcessObserver,
    report: PhaseReporter,
) -> EncodingJobResult:
    """Encode, prove recovery with the matching decoder, then publish the PNG."""
    report("processing")
    output = scratch / "encoded.png"
    run_package(
        package_root / "encoder",
        ["encode", "--image", str(image), "--output", str(output)],
        secret_input=secret_input(
            {"password": secrets["password"], "message": secrets["message"]}
        ),
        directory=scratch,
        on_process=on_process,
    )
    report("verifying_saved_image")
    try:
        restored = run_package(
            package_root / "decoder",
            ["decode", "--image", str(output)],
            secret_input=secret_input({"password": secrets["password"]}),
            directory=scratch,
            on_process=on_process,
        )
    except ApplicationFailure as failure:
        if failure.code == "message_recovery_failed":
            raise verification_failure() from None
        raise
    if restored != secrets["message"].encode("utf-8", errors="strict"):
        raise verification_failure()
    published = inference.files.publish_result(output)
    return EncodingJobResult(
        artifact_identifier=published.image_reference,
        filename=f"stegolab-encoded-{published.image_reference[8:16]}.png",
        width=published.summary.prepared_width,
        height=published.summary.prepared_height,
        png_bytes=published.stored_bytes,
        message_byte_count=record.message_byte_count,
    )


def decode_image(
    inference: "InferenceServices",
    identifier: str,
    package_root: Path,
    image: Path,
    scratch: Path,
    secrets: dict[str, str],
    on_process: ProcessObserver,
    report: PhaseReporter,
) -> DecodingJobResult:
    """Recover authenticated text into memory only, or fail with the fixed message."""
    report("processing")
    restored = run_package(
        package_root / "decoder",
        ["decode", "--image", str(image)],
        secret_input=secret_input({"password": secrets["password"]}),
        directory=scratch,
        on_process=on_process,
    )
    if len(restored) > MAXIMUM_MESSAGE_BYTES:
        raise export_failure()
    try:
        text = restored.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise export_failure() from None
    entry = inference.texts.put(identifier, text)
    return DecodingJobResult(
        text_byte_count=entry.byte_count, expires_at=entry.expires_at
    )


def run_inference(
    service: "WorkspaceJobService", identifier: str, record: InferenceJobRecord
) -> None:
    """Take the job's secrets once, run the packages, and publish the result."""
    inference = require_inference(service)
    secrets = inference.secrets.pop(identifier)
    if secrets is None:
        raise ApplicationFailure(INPUTS_LOST["code"], INPUTS_LOST["message"], 422)

    def remember_process(process: subprocess.Popen[bytes] | None) -> None:
        """Track the package process so shutdown can stop it."""
        service.process = process

    def report(phase: str) -> None:
        """Publish the current step, refusing new processes once shutdown began."""
        if service.closing.is_set():
            raise ApplicationFailure(
                "service_stopping", "The application is stopping. Retry after restart."
            )
        with service.lock:
            service._change(service.store.get(identifier), phase=phase)

    try:
        _, package_root = inference.installed.resolve(record.model_identifier)
        verify_model_pair(package_root)
        image = inference.files.image_path(record.image_reference)
        with tempfile.TemporaryDirectory(
            prefix="job-", dir=inference.files.scratch
        ) as name:
            scratch = Path(name)
            result: EncodingJobResult | DecodingJobResult
            if record.operation == "encode":
                result = encode_image(
                    inference,
                    package_root,
                    image,
                    scratch,
                    secrets,
                    record,
                    remember_process,
                    report,
                )
            else:
                result = decode_image(
                    inference,
                    identifier,
                    package_root,
                    image,
                    scratch,
                    secrets,
                    remember_process,
                    report,
                )
    finally:
        secrets.clear()
    with service.lock:
        service._change(
            service.store.get(identifier),
            status="completed",
            phase="completed",
            available_actions=[],
            result=result.model_dump(mode="json"),
            progress=1.0,
            error=None,
        )
