"""Keep requested CUDA verification honest without executing a GPU locally."""

import os
from pathlib import Path
from typing import Literal

import pytest
import torch
from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.model_export_devices import configure_cuda_numerics
from backend_service.model_exports import export_models
from backend_service.model_networks import build_models, model_pair_identifier
from schemas.model_exports import ExportMetadata, ExportVerification


def test_isolated_runtime_uses_training_fp32_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set the independent process policy without initializing CUDA hardware."""
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    try:
        configure_cuda_numerics()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
        with pytest.raises(ApplicationFailure) as caught:
            configure_cuda_numerics()
        assert caught.value.code == "cuda_numerical_environment"
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]
        torch.backends.cudnn.benchmark = previous[3]
        torch.backends.cudnn.deterministic = previous[4]


@pytest.mark.parametrize("cuda_failure", [False, True])
def test_requested_cuda_verification_gates_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda_failure: bool,
) -> None:
    """CPU parity alone cannot publish a package requested for CUDA verification."""
    encoder, decoder = build_models(9)
    identifier = model_pair_identifier(encoder, decoder)
    calls: list[str] = []
    destination = tmp_path / "pair"

    def package(
        model: nn.Module,
        directory: Path,
        role: Literal["encoder", "decoder"],
        metadata: ExportMetadata,
        deadline: float | None,
        runtime_dependencies: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Isolate orchestration from the separately covered graph serialization."""
        directory.mkdir()

    def verify(
        first: nn.Module,
        second: nn.Module,
        directory: Path,
        *,
        deadline: float | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> ExportVerification:
        """Fail the CUDA check after successful CPU parity, before publication."""
        assert not destination.exists()
        calls.append(device)
        if device == "cuda" and cuda_failure:
            raise ApplicationFailure("cuda_fixture_failure", "GPU check failed.")
        return ExportVerification(
            compatibility_identifier=identifier,
            cases_checked=6,
            maximum_absolute_difference=0.0,
            device=device,
        )

    monkeypatch.setattr("backend_service.model_exports._package", package)
    monkeypatch.setattr(
        "backend_service.model_export_verification.verify_exports", verify
    )
    metadata = ExportMetadata(
        compatibility_identifier=identifier, source_identifier="public_device_fixture"
    )
    if cuda_failure:
        with pytest.raises(ApplicationFailure):
            export_models(
                encoder, decoder, destination, metadata, verification_device="cuda"
            )
        assert not destination.exists()
        assert not list(tmp_path.glob(".stegolab-export-*"))
    else:
        export_models(
            encoder, decoder, destination, metadata, verification_device="cuda"
        )
        report = ExportVerification.model_validate_json(
            (destination / "verification.json").read_bytes()
        )
        assert report.device == "cuda"
        assert (destination / "verification-cpu.json").is_file()
    assert calls == ["cpu", "cuda"]
