"""Saved-PNG pilot trials with explicit CPU and CUDA tensor boundaries."""

import time

import torch
from torch import Tensor, nn

from backend_service.dataset_reader import DatasetExample
from backend_service.failures import ApplicationFailure
from backend_service.message_protocol import decode_message, encode_message
from backend_service.model_data import load_center_crop
from backend_service.model_evaluation_images import (
    PUBLIC_FIXTURE_PASSWORD,
    EvaluationDeadline,
    _saved_pixels,
    check_deadline,
    validate_model_tensor,
)
from backend_service.model_networks import quantize_image
from backend_service.model_quality import image_quality
from backend_service.payload_capacity import calculate_capacity
from backend_service.payload_map import create_payload_map, recover_payload_bytes
from backend_service.pilot_runtime import synchronize_device
from schemas.evaluation import PngEvaluationTrial
from schemas.protocol import ProtocolContext


def _cpu_output(values: Tensor, shape: tuple[int, ...], device: torch.device) -> Tensor:
    """Validate placement before transferring a finite model result to CPU."""
    if not isinstance(values, Tensor) or values.device != device:
        raise ApplicationFailure(
            "model_output_invalid",
            "The model returned values on an unexpected device.",
            422,
        )
    result = values.detach().cpu()
    validate_model_tensor(result, shape)
    return result


def run_pilot_png_trial(
    encoder: nn.Module,
    decoder: nn.Module,
    example: DatasetExample,
    compatibility_identifier: str,
    *,
    device: torch.device,
    deadline: float,
    measure_quality: bool = True,
) -> PngEvaluationTrial:
    """Attempt a full public message once, preserving input and recovery failures."""
    started = time.monotonic()
    source_identity = example.record.source_identity or example.record.source_path
    context = ProtocolContext(
        width=1024, height=1024, compatibility_identifier=compatibility_identifier
    )
    error_code: str | None = None
    bit_error: float | None = None
    peak_ratio: float | None = None
    similarity: float | None = None
    clipped: float | None = None
    identical = recovered = False
    try:
        check_deadline(deadline)
        cover = load_center_crop(example, 1024, 1024).unsqueeze(0)
        check_deadline(deadline)
        message = "M" * calculate_capacity(context).maximum_message_bytes
        protected = encode_message(message, PUBLIC_FIXTURE_PASSWORD, context)
        payload = (
            torch.from_numpy(create_payload_map(protected, context))
            .float()
            .unsqueeze(0)
        )
        check_deadline(deadline)
        raw = _cpu_output(
            encoder(cover.to(device), payload.to(device)), (1, 3, 1024, 1024), device
        )
        clipped = (
            float(((raw < 0.0) | (raw > 1.0)).float().mean())
            if measure_quality
            else None
        )
        check_deadline(deadline)
        reopened = _saved_pixels(quantize_image(raw))
        check_deadline(deadline)
        logits = _cpu_output(decoder(reopened.to(device)), (1, 1, 1024, 1024), device)
        synchronize_device(device)
        check_deadline(deadline)
        bit_error = float(((logits > 0) != payload.bool()).float().mean())
        if measure_quality:
            peak_ratio, similarity, identical = image_quality(cover, reopened)
        check_deadline(deadline)
        restored = recover_payload_bytes(logits[0].contiguous().numpy(), context)
        recovered = (
            decode_message(restored, PUBLIC_FIXTURE_PASSWORD, context) == message
        )
    except EvaluationDeadline:
        error_code = "evaluation_deadline"
    except ApplicationFailure as failure:
        error_code = failure.code
    except (RuntimeError, OSError, ValueError, MemoryError):
        error_code = "pilot_evaluation_failed"
    return PngEvaluationTrial(
        source_identity=source_identity,
        width=1024,
        height=1024,
        fixture_kind="maximum",
        recovered=recovered,
        raw_bit_error_rate=bit_error,
        peak_signal_to_noise_ratio=peak_ratio,
        structural_similarity=similarity,
        identical_pixels=identical,
        clipped_fraction=clipped,
        elapsed_seconds=time.monotonic() - started,
        error_code=error_code,
    )
