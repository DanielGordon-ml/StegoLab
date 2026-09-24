"""Exercise evaluation accounting using an explicit non-neural test channel."""

import time
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from backend_service import model_evaluation
from backend_service.dataset_reader import DatasetExample
from backend_service.failures import ApplicationFailure
from backend_service.model_data import ProofData
from backend_service.model_evaluation import evaluate_models
from backend_service.model_evaluation_images import run_png_trial
from backend_service.model_quality import image_quality
from schemas.datasets import DatasetImageRecord, DatasetManifest
from schemas.evaluation import PngEvaluationTrial


class OracleEncoder(nn.Module):
    """Use a deliberately visible parity channel only for pipeline testing."""

    def __init__(self) -> None:
        """Include normalization solely to detect accidental evaluation updates."""
        super().__init__()
        self.normalization = nn.BatchNorm2d(3)

    def forward(self, cover: Tensor, payload: Tensor) -> Tensor:
        """Write payload bits into the red integer channel, without learned weights."""
        self.normalization(cover)
        red = torch.round(cover[:, :1] * 255)
        red = (red - red.remainder(2) + payload) / 255
        return torch.cat((red, cover[:, 1:]), dim=1)


class OracleDecoder(nn.Module):
    """Read the test parity channel without a cover, message, or encoder reference."""

    def forward(self, image: Tensor) -> Tensor:
        """Convert red-channel integer parity to finite unambiguous logits."""
        return (torch.round(image[:, :1] * 255).remainder(2) * 2 - 1) * 8


def _proof_data() -> ProofData:
    """Build an in-memory handoff; file reads are replaced only in suite tests."""
    examples = tuple(
        DatasetExample(
            DatasetImageRecord.model_construct(),  # type: ignore[call-arg]
            Path("unused"),
        )
        for _ in range(4)
    )
    crops = tuple(torch.zeros((3, 256, 256)) for _ in range(4))
    return ProofData(
        crops,
        crops,
        DatasetManifest.model_construct(revision="a" * 64),  # type: ignore[call-arg]
        {"train": ("a", "b", "c", "d"), "tuning": ("e", "f", "g", "h")},
        examples,
    )


def test_real_png_channel_authenticates_odd_full_capacity_fixture() -> None:
    """Test actual encryption and disk PNG reopening independently of model learning."""
    cover = torch.full((1, 3, 517, 513), 128 / 255)
    with torch.inference_mode():
        trial = run_png_trial(
            OracleEncoder(),
            OracleDecoder(),
            cover,
            "public_test_cover",
            "public_test_model",
            "maximum",
            None,
        )
    assert trial.recovered
    assert trial.raw_bit_error_rate == 0.0
    assert trial.peak_signal_to_noise_ratio is not None
    assert trial.structural_similarity is not None
    assert trial.error_code is None
    serialized = trial.model_dump_json()
    assert "password" not in serialized
    assert "MMMM" not in serialized


def test_full_suite_counts_every_attempt_and_preserves_training_randomness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All identities, sizes, and messages count; a failed attempt fails the gate."""
    data = _proof_data()
    encoder, decoder = OracleEncoder(), OracleDecoder()
    decoder.eval()
    initial_random_state = torch.get_rng_state().clone()
    calls: list[tuple[str, int, int, str]] = []

    def crop(example: DatasetExample, width: int, height: int) -> Tensor:
        """Return an inexpensive correctly shaped stand-in for already-tested reads."""
        return torch.zeros((3, height, width))

    def trial(
        first: nn.Module,
        second: nn.Module,
        cover: Tensor,
        identity: str,
        compatibility: str,
        kind: str,
        deadline: float | None,
    ) -> PngEvaluationTrial:
        """Record scheduling while emulating one declared recovery failure."""
        assert not first.training and not second.training
        height, width = cover.shape[-2:]
        calls.append((identity, width, height, kind))
        return PngEvaluationTrial.model_validate(
            {
                "source_identity": identity,
                "width": width,
                "height": height,
                "fixture_kind": kind,
                "recovered": len(calls) != 2,
                "raw_bit_error_rate": 0.0,
                "peak_signal_to_noise_ratio": 41.0,
                "structural_similarity": 0.99,
                "clipped_fraction": 0.0,
                "elapsed_seconds": 0.0,
            }
        )

    monkeypatch.setattr(model_evaluation, "load_center_crop", crop)
    monkeypatch.setattr(model_evaluation, "run_png_trial", trial)
    report = evaluate_models(encoder, decoder, data, "public_test_model")
    assert report.completed and report.bit_cases_completed == 16
    assert report.raw_bit_error_rate == 0.0
    assert len(calls) == len(set(calls)) == 36
    assert report.exact_recovery_count == 35
    assert report.exact_recovery_rate == 35 / 36
    assert not report.learning_gate_passed
    assert encoder.training and not decoder.training
    assert encoder.normalization.num_batches_tracked is not None
    assert int(encoder.normalization.num_batches_tracked) == 0
    assert torch.equal(initial_random_state, torch.get_rng_state())
    calls.clear()
    validation = evaluate_models(
        encoder, decoder, data, "public_test_model", lightweight=True
    )
    assert validation.completed and validation.expected_bit_cases == 0
    assert len(calls) == 4
    assert all(item[1:] == (512, 512, "maximum") for item in calls)


def test_expired_budget_is_incomplete_and_restores_modes() -> None:
    """Do no inference after expiry and never convert missing cases to success."""
    encoder, decoder = OracleEncoder(), OracleDecoder()
    report = evaluate_models(
        encoder,
        decoder,
        _proof_data(),
        "public_test_model",
        deadline=time.monotonic() - 1.0,
    )
    assert report.stopped_reason == "deadline"
    assert not report.completed and not report.learning_gate_passed
    assert report.bit_cases_completed == 0 and report.png_trials == []
    assert encoder.training and decoder.training


def test_identical_image_quality_has_serializable_perfect_score() -> None:
    """Represent infinite PSNR explicitly while retaining exact SSIM and equality."""
    cover = torch.zeros((1, 3, 15, 17))
    peak_ratio, similarity, identical = image_quality(cover, cover)
    assert peak_ratio is None and similarity == 1.0 and identical


@pytest.mark.parametrize(
    "failure_code",
    ["model_output_invalid", "image_write", "protocol_resources_unavailable"],
)
def test_invalid_outputs_or_resources_abort_validation_without_completion(
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str,
) -> None:
    """Keep the failed trial and distinguish operational errors from recovery misses."""

    class FailingEncoder(nn.Module):
        """Generate a real invalid tensor or the selected safe operational failure."""

        def forward(self, cover: Tensor, payload: Tensor) -> Tensor:
            """Return nonfinite RGB or raise a fixed resource failure."""
            if failure_code == "model_output_invalid":
                return torch.full_like(cover, float("nan"))
            raise ApplicationFailure(failure_code, "Public operational failure.")

    def crop(example: DatasetExample, width: int, height: int) -> Tensor:
        """Keep this test focused on error classification after source loading."""
        return torch.zeros((3, height, width))

    monkeypatch.setattr(model_evaluation, "load_center_crop", crop)
    encoder = FailingEncoder()
    report = evaluate_models(
        encoder, OracleDecoder(), _proof_data(), "public_test_model", lightweight=True
    )
    assert report.stopped_reason == "failed" and not report.completed
    assert len(report.png_trials) == 1
    assert report.png_trials[0].error_code == failure_code
    assert report.exact_recovery_count == 0 and encoder.training
