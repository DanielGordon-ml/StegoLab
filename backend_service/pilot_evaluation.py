"""Frozen tuning measurements without held-out access or success filtering."""

import math
import statistics
import time
from typing import Literal

import torch
from torch import nn

from backend_service.model_quality import peak_process_memory_bytes
from backend_service.pilot_data import PilotData
from backend_service.pilot_evaluation_images import run_pilot_png_trial
from schemas.evaluation import PngEvaluationTrial
from schemas.pilot_training import PilotEvaluationReport


def _median(values: list[float]) -> float | None:
    """Represent missing or mathematically infinite PSNR without invalid JSON."""
    if not values:
        return None
    result = statistics.median(values)
    return result if math.isfinite(result) else None


def evaluate_pilot_models(
    encoder: nn.Module,
    decoder: nn.Module,
    data: PilotData,
    compatibility_identifier: str,
    *,
    device: torch.device,
    deadline: float,
    smoke: bool = False,
) -> PilotEvaluationReport:
    """Run thirty-two tuning covers or one unranked CPU protocol smoke case."""
    started = time.monotonic()
    expected = 1 if smoke else 32
    trials: list[PngEvaluationTrial] = []
    flags = [
        (module, module.training)
        for model in (encoder, decoder)
        for module in model.modules()
    ]
    reason: Literal["complete", "deadline", "failed"] = "complete"
    try:
        encoder.eval()
        decoder.eval()
        with torch.inference_mode():
            for example in data.tuning_examples[:expected]:
                if time.monotonic() >= deadline:
                    reason = "deadline"
                    break
                trial = run_pilot_png_trial(
                    encoder,
                    decoder,
                    example,
                    compatibility_identifier,
                    device=device,
                    deadline=deadline,
                    measure_quality=not smoke,
                )
                trials.append(trial)
                if trial.error_code == "evaluation_deadline":
                    reason = "deadline"
                    break
                if trial.error_code not in (None, "message_recovery_failed"):
                    reason = "failed"
    finally:
        for module, flag in flags:
            module.training = flag
    complete = reason == "complete" and len(trials) == expected
    if len(trials) != expected and reason == "complete":
        reason = "failed"
    recovered = sum(trial.recovered for trial in trials)
    ratios = [
        math.inf if trial.identical_pixels else trial.peak_signal_to_noise_ratio
        for trial in trials
        if trial.identical_pixels or trial.peak_signal_to_noise_ratio is not None
    ]
    similarities = [
        trial.structural_similarity
        for trial in trials
        if trial.structural_similarity is not None
    ]
    clipped = [
        trial.clipped_fraction for trial in trials if trial.clipped_fraction is not None
    ]
    return PilotEvaluationReport(
        kind="cpu_smoke" if smoke else "tuning",
        compatibility_identifier=compatibility_identifier,
        dataset_revision=data.manifest.revision,
        selection_identifier=data.selection.selection_checksum,
        completed=complete,
        expected_png_trials=expected,
        png_trials=trials,
        exact_recovery_count=recovered,
        exact_recovery_rate=recovered / len(trials) if trials else None,
        median_peak_signal_to_noise_ratio=_median(
            [value for value in ratios if value is not None]
        ),
        median_structural_similarity=_median(similarities),
        mean_clipped_fraction=statistics.mean(clipped) if clipped else None,
        elapsed_seconds=time.monotonic() - started,
        peak_process_memory_bytes=peak_process_memory_bytes(),
        peak_device_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        quality_measured=not smoke,
        stopped_reason=reason,
    )
