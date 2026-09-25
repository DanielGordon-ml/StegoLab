"""Command, environment, and failure mapping shared by isolated package runs."""

import os
import sys
from pathlib import Path

from backend_service.failures import ApplicationFailure
from backend_service.model_export_io import export_failure
from backend_service.protocol_failures import RECOVERY_MESSAGE, recovery_failure

PACKAGE_BOOTSTRAP = (
    "import runpy,sys; package=sys.argv.pop(1); "
    "sys.path.insert(0,package); "
    "runpy.run_path(package+'/runtime.py',run_name='__main__')"
)


def package_command(package: Path, arguments: list[str]) -> list[str]:
    """Start the bundled entry point with no repository path or site packages."""
    return [
        sys.executable,
        "-I",
        "-B",
        "-X",
        "utf8",
        "-c",
        PACKAGE_BOOTSTRAP,
        str(package.resolve()),
        *arguments,
    ]


def package_environment() -> dict[str, str]:
    """Copy the environment without inherited Python paths and with one thread."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    return environment


def classify_package_failure(error_output: bytes) -> ApplicationFailure:
    """Map only the exact frozen recovery line to a recovery miss; hide the rest."""
    if error_output == (RECOVERY_MESSAGE + "\n").encode("utf-8"):
        return recovery_failure()
    return export_failure()
