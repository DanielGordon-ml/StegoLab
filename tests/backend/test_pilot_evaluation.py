"""Keep every tuning attempt and exercise the real PNG protocol boundary."""

import time
from pathlib import Path

import pytest
import torch
from pilot_data_fixtures import pilot_revision
from torch import Tensor, nn

from backend_service.dataset_reader import DatasetExample
from backend_service.pilot_data import load_pilot_data
from backend_service.pilot_evaluation import evaluate_pilot_models
from backend_service.pilot_evaluation_images import run_pilot_png_trial
from schemas.evaluation import PngEvaluationTrial


class PublicEncoder(nn.Module):
    """Put public fixture bits in a visible channel; this is not a learned model."""

    def forward(self, cover: Tensor, payload: Tensor) -> Tensor:
        """Keep two channels and replace the first with protocol fixture bits."""
        return torch.cat((payload, cover[:, 1:]), dim=1)


class PublicDecoder(nn.Module):
    """Recover the deliberately visible fixture channel as logits."""

    def forward(self, image: Tensor) -> Tensor:
        """Return a signed decision from the first integer PNG channel."""
        return image[:, :1] * 2 - 1


def test_cpu_smoke_saved_png_recovers_without_quality_measurement(
    tmp_path: Path,
) -> None:
    """One public functional case measures no scientific quality or release gate."""
    torch.set_num_threads(1)
    data = load_pilot_data(pilot_revision(tmp_path / "dataset"))
    report = evaluate_pilot_models(
        PublicEncoder(),
        PublicDecoder(),
        data,
        "public_fixture",
        device=torch.device("cpu"),
        deadline=time.monotonic() + 60,
        smoke=True,
    )
    assert report.completed and report.exact_recovery_count == 1
    assert report.expected_png_trials == 1 and len(report.png_trials) == 1
    assert not report.quality_measured and not report.release_qualified
    assert report.median_peak_signal_to_noise_ratio is None
    assert report.median_structural_similarity is None


def test_all_32_attempts_remain_and_modes_are_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure stays in its original position and is not retried or replaced."""
    data = load_pilot_data(pilot_revision(tmp_path / "dataset", tuning_count=34))
    calls: list[str] = []

    def trial(
        encoder: nn.Module,
        decoder: nn.Module,
        example: DatasetExample,
        compatibility_identifier: str,
        **keywords: object,
    ) -> PngEvaluationTrial:
        """Return a deterministic mixture without running model quality work."""
        identity = example.record.source_identity or example.record.source_path
        calls.append(identity)
        good = len(calls) % 2 == 0
        return PngEvaluationTrial(
            source_identity=identity,
            width=1024,
            height=1024,
            fixture_kind="maximum",
            recovered=good,
            elapsed_seconds=0,
            error_code=None if good else "message_recovery_failed",
            peak_signal_to_noise_ratio=40 if good else 20,
            structural_similarity=0.99 if good else 0.8,
        )

    monkeypatch.setattr("backend_service.pilot_evaluation.run_pilot_png_trial", trial)
    encoder, decoder = PublicEncoder(), PublicDecoder()
    encoder.train()
    decoder.eval()
    report = evaluate_pilot_models(
        encoder,
        decoder,
        data,
        "public_fixture",
        device=torch.device("cpu"),
        deadline=time.monotonic() + 60,
    )
    assert report.completed and len(report.png_trials) == 32
    assert report.exact_recovery_count == 16 and report.exact_recovery_rate == 0.5
    assert report.median_peak_signal_to_noise_ratio == 30
    assert tuple(calls) == data.identities["tuning"]
    assert encoder.training and not decoder.training
    assert not report.release_qualified


def test_deadline_and_bad_cover_are_reported(tmp_path: Path) -> None:
    """Never turn missing inputs or an expired run into an apparent full suite."""
    data = load_pilot_data(pilot_revision(tmp_path / "dataset"))
    encoder, decoder = PublicEncoder(), PublicDecoder()
    report = evaluate_pilot_models(
        encoder,
        decoder,
        data,
        "public_fixture",
        device=torch.device("cpu"),
        deadline=0,
    )
    assert not report.completed and not report.png_trials
    assert report.stopped_reason == "deadline"
    example = data.tuning_examples[0]
    example.path.write_bytes(b"changed")
    trial = run_pilot_png_trial(
        encoder,
        decoder,
        example,
        "public_fixture",
        device=torch.device("cpu"),
        deadline=time.monotonic() + 60,
    )
    assert not trial.recovered and trial.error_code == "model_dataset_invalid"
