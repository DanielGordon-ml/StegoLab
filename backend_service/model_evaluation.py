"""Bounded CPU model scoring, independent validation randomness, and honest gates."""

import math
import statistics
import time
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.model_data import ProofData, load_center_crop
from backend_service.model_evaluation_images import (
    EvaluationDeadline,
    FixtureKind,
    check_deadline,
    run_png_trial,
    validate_model_tensor,
)
from backend_service.model_networks import quantize_image
from backend_service.model_quality import peak_process_memory_bytes
from schemas.evaluation import EvaluationReport, PngEvaluationTrial


def _median(values: Sequence[float]) -> float | None:
    """Use null for missing or perfect/infinite PSNR, never invalid JSON numbers."""
    if not values:
        return None
    value = statistics.median(values)
    return value if math.isfinite(value) else None


def _bit_case(
    encoder: nn.Module,
    decoder: nn.Module,
    data: ProofData,
    index: int,
    deadline: float | None,
) -> float:
    """Score a fixed independent random map without consuming training randomness."""
    generator = torch.Generator(device="cpu").manual_seed(1 + index)
    payload = torch.randint(0, 2, (1, 1, 256, 256), generator=generator).float()
    cover = data.tuning_crops[index // 4].unsqueeze(0)
    raw = encoder(cover, payload)
    validate_model_tensor(raw, (1, 3, 256, 256))
    check_deadline(deadline)
    logits = decoder(quantize_image(raw))
    validate_model_tensor(logits, (1, 1, 256, 256))
    return float(((logits > 0) != payload.bool()).float().mean())


def _make_report(
    data: ProofData,
    compatibility_identifier: str,
    lightweight: bool,
    errors: list[float],
    trials: list[PngEvaluationTrial],
    reason: Literal["complete", "deadline", "failed"],
    start: float,
) -> EvaluationReport:
    """Aggregate every attempt without filtering recovery failures or retries."""
    expected_bits, expected_images = (0, 4) if lightweight else (16, 36)
    complete = (
        reason == "complete"
        and len(errors) == expected_bits
        and len(trials) == expected_images
    )
    recovered = sum(trial.recovered for trial in trials)
    bit_error = statistics.mean(errors) if errors else None
    peak_ratios = [
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
    return EvaluationReport(
        kind="validation" if lightweight else "proof",
        compatibility_identifier=compatibility_identifier,
        dataset_revision=data.manifest.revision,
        completed=complete,
        expected_bit_cases=expected_bits,
        bit_cases_completed=len(errors),
        raw_bit_error_rate=bit_error,
        expected_png_trials=expected_images,
        png_trials=trials,
        exact_recovery_count=recovered,
        exact_recovery_rate=recovered / len(trials) if trials else None,
        median_peak_signal_to_noise_ratio=_median(
            [value for value in peak_ratios if value is not None]
        ),
        median_structural_similarity=_median(similarities),
        mean_clipped_fraction=statistics.mean(clipped) if clipped else None,
        elapsed_seconds=time.monotonic() - start,
        peak_process_memory_bytes=peak_process_memory_bytes(),
        learning_gate_passed=(
            not lightweight
            and complete
            and bit_error is not None
            and bit_error <= 0.1
            and recovered == 36
        ),
        stopped_reason=reason,
    )


def evaluate_models(
    encoder: nn.Module,
    decoder: nn.Module,
    data: ProofData,
    compatibility_identifier: str,
    *,
    deadline: float | None = None,
    lightweight: bool = False,
) -> EvaluationReport:
    """Measure proof or four-image validation and always restore original modes."""
    start = time.monotonic()
    flags = [
        (module, module.training)
        for model in (encoder, decoder)
        for module in model.modules()
    ]
    errors: list[float] = []
    trials: list[PngEvaluationTrial] = []
    reason: Literal["complete", "deadline", "failed"] = "complete"
    sizes = ((512, 512),) if lightweight else ((512, 512), (513, 517), (1024, 1024))
    kinds: tuple[FixtureKind, ...] = (
        ("maximum",) if lightweight else ("empty", "unicode", "maximum")
    )
    try:
        encoder.eval()
        decoder.eval()
        with torch.inference_mode():
            for index in range(0 if lightweight else 16):
                check_deadline(deadline)
                errors.append(_bit_case(encoder, decoder, data, index, deadline))
                check_deadline(deadline)
            for index, example in enumerate(data.tuning_examples):
                for width, height in sizes:
                    check_deadline(deadline)
                    cover = load_center_crop(example, width, height).unsqueeze(0)
                    for kind in kinds:
                        check_deadline(deadline)
                        trial = run_png_trial(
                            encoder,
                            decoder,
                            cover,
                            data.identities["tuning"][index],
                            compatibility_identifier,
                            kind,
                            deadline,
                        )
                        trials.append(trial)
                        if trial.error_code == "evaluation_deadline":
                            raise EvaluationDeadline
                        if trial.error_code not in (None, "message_recovery_failed"):
                            reason = "failed"
                            return _make_report(
                                data,
                                compatibility_identifier,
                                lightweight,
                                errors,
                                trials,
                                reason,
                                start,
                            )
    except EvaluationDeadline:
        reason = "deadline"
    except (ApplicationFailure, RuntimeError, OSError, ValueError, MemoryError):
        reason = "failed"
    finally:
        for module, flag in flags:
            module.training = flag
    return _make_report(
        data, compatibility_identifier, lightweight, errors, trials, reason, start
    )
