"""Run one installed package process with secrets on stdin and a process handle."""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from backend_service.failures import ApplicationFailure
from backend_service.model_export_io import export_failure
from backend_service.model_export_process import (
    classify_package_failure,
    package_command,
    package_environment,
)

PROCESS_TIMEOUT_SECONDS = 120.0
MAXIMUM_OUTPUT_BYTES = 64 * 1024
MAXIMUM_SECRET_INPUT_BYTES = 16_384
ProcessObserver = Callable[[subprocess.Popen[bytes] | None], None]


def timeout_failure() -> ApplicationFailure:
    """Explain the fixed time limit of one model process."""
    return ApplicationFailure(
        "inference_timeout",
        "The model did not finish within two minutes. Try a smaller image.",
        422,
    )


def secret_input(values: dict[str, str]) -> bytes:
    """Serialize secrets for the package's protected standard input only."""
    content = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(content) > MAXIMUM_SECRET_INPUT_BYTES:
        raise export_failure()
    return content


def run_package(
    package: Path,
    arguments: list[str],
    *,
    secret_input: bytes,
    directory: Path,
    on_process: ProcessObserver,
    timeout: float = PROCESS_TIMEOUT_SECONDS,
) -> bytes:
    """Run a package entry point in isolation and return its bounded output."""
    try:
        process = subprocess.Popen(
            package_command(package, arguments),
            cwd=directory,
            env=package_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError:
        raise export_failure() from None
    on_process(process)
    try:
        try:
            output, errors = process.communicate(input=secret_input, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise timeout_failure() from None
        except OSError:
            process.kill()
            process.communicate()
            raise export_failure() from None
    finally:
        on_process(None)
    if process.returncode:
        raise classify_package_failure(errors)
    if len(output) > MAXIMUM_OUTPUT_BYTES:
        raise export_failure()
    return output
