"""Distinguish authenticated recovery misses from failed independent runtimes."""

import subprocess
from pathlib import Path

import pytest

from backend_service.failures import ApplicationFailure
from backend_service.model_export_verification import run_export_process
from backend_service.protocol_failures import RECOVERY_MESSAGE


@pytest.mark.parametrize(
    ("error_output", "expected_code"),
    [
        ((RECOVERY_MESSAGE + "\n").encode(), "message_recovery_failed"),
        (b"public-review-sentinel", "model_export_unavailable"),
        (
            (RECOVERY_MESSAGE + "\nextra diagnostic\n").encode(),
            "model_export_unavailable",
        ),
        (b"", "model_export_unavailable"),
    ],
)
def test_export_process_classifies_only_exact_recovery_failure(
    error_output: bytes,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Expose only a fixed recovery miss and never include child diagnostics."""

    def failed_child(
        *arguments: object, **keywords: object
    ) -> subprocess.CompletedProcess[bytes]:
        """Return controlled public stderr without starting a subprocess."""
        return subprocess.CompletedProcess([], 1, b"", error_output)

    monkeypatch.setattr(
        "backend_service.model_export_verification.subprocess.run", failed_child
    )
    with pytest.raises(ApplicationFailure) as failure:
        run_export_process(tmp_path / "decoder", ["decode"], directory=tmp_path)
    assert failure.value.code == expected_code
    assert "public-review-sentinel" not in str(failure.value)
