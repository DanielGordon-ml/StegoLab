"""Run package processes with stdin secrets, bounded output, and a time limit."""

import subprocess
from pathlib import Path

import pytest

from backend_service.failures import ApplicationFailure
from backend_service.inference_execution import run_package, secret_input
from backend_service.model_export_process import classify_package_failure
from backend_service.protocol_failures import RECOVERY_MESSAGE


def fake_package(directory: Path, body: str) -> Path:
    """Write a stand-in runtime so no model graph is needed for process checks."""
    package = directory / "package"
    package.mkdir()
    (package / "runtime.py").write_text(body, encoding="utf-8")
    return package


def test_stdin_secrets_reach_the_package_and_output_is_returned(
    tmp_path: Path,
) -> None:
    """Deliver secrets only through standard input and hand back standard output."""
    package = fake_package(
        tmp_path,
        "import sys\nsys.stdout.write(sys.stdin.read()[::-1])\n",
    )
    seen: list[subprocess.Popen[bytes] | None] = []
    output = run_package(
        package,
        ["decode"],
        secret_input=secret_input({"password": "sentinel-secret"}),
        directory=tmp_path,
        on_process=seen.append,
    )
    assert output == b'{"password":"sentinel-secret"}'[::-1]
    assert len(seen) == 2 and seen[0] is not None and seen[1] is None
    assert seen[0].returncode == 0
    assert "sentinel-secret" not in str(seen[0].args)


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (
            "import sys\nsys.stderr.write("
            + repr(RECOVERY_MESSAGE + "\n")
            + ")\nraise SystemExit(1)\n",
            "message_recovery_failed",
        ),
        (
            "import sys\nsys.stderr.write('public-review-sentinel')\n"
            "raise SystemExit(1)\n",
            "model_export_unavailable",
        ),
        (
            "import sys\nsys.stdout.write('x' * 70000)\n",
            "model_export_unavailable",
        ),
        ("import time\ntime.sleep(5)\n", "inference_timeout"),
    ],
)
def test_failures_are_fixed_and_never_include_child_output(
    tmp_path: Path, body: str, code: str
) -> None:
    """Map the exact recovery line, hide other diagnostics, and enforce the limit."""
    package = fake_package(tmp_path, body)
    with pytest.raises(ApplicationFailure) as failure:
        run_package(
            package,
            ["decode"],
            secret_input=b"{}",
            directory=tmp_path,
            on_process=lambda process: None,
            timeout=1.0,
        )
    assert failure.value.code == code
    assert "public-review-sentinel" not in str(failure.value)


def test_secret_input_is_bounded_and_classification_is_exact() -> None:
    """Keep stdin payloads small and treat any extra stderr text as a runtime fault."""
    with pytest.raises(ApplicationFailure):
        secret_input({"message": "m" * 20_000, "password": "p"})
    assert classify_package_failure(b"").code == "model_export_unavailable"
    assert (
        classify_package_failure((RECOVERY_MESSAGE + "\nmore\n").encode()).code
        == "model_export_unavailable"
    )
    assert (
        classify_package_failure((RECOVERY_MESSAGE + "\n").encode()).code
        == "message_recovery_failed"
    )
