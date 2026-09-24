"""Public fixture trials through the actual encrypted, saved-PNG channel."""

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from backend_service.failures import ApplicationFailure
from backend_service.image_output import write_png
from backend_service.message_protocol import decode_message, encode_message
from backend_service.model_networks import quantize_image
from backend_service.model_quality import image_quality
from backend_service.payload_capacity import calculate_capacity
from backend_service.payload_map import create_payload_map, recover_payload_bytes
from schemas.evaluation import PngEvaluationTrial
from schemas.protocol import ProtocolContext

FixtureKind = Literal["empty", "unicode", "maximum"]
PUBLIC_FIXTURE_PASSWORD = "StegoLab public model proof password"
PUBLIC_UNICODE_MESSAGE = "StegoLab שלום 🌍\ne\u0301\x00"


class EvaluationDeadline(Exception):
    """End bounded evaluation without treating incomplete work as success."""


def check_deadline(deadline: float | None) -> None:
    """Check the caller's absolute monotonic deadline between bounded operations."""
    if deadline is not None and time.monotonic() >= deadline:
        raise EvaluationDeadline


def validate_model_tensor(values: Tensor, shape: tuple[int, ...]) -> None:
    """Reject invalid model outputs before pixel conversion or metric arithmetic."""
    if (
        values.dtype != torch.float32
        or values.device.type != "cpu"
        or tuple(values.shape) != shape
        or not bool(torch.isfinite(values).all())
    ):
        raise ApplicationFailure(
            "model_output_invalid", "The model returned unsupported image values.", 422
        )


def _saved_pixels(values: Tensor) -> Tensor:
    """Write and reopen one real PNG while keeping no permanent proof images."""
    pixels = (
        torch.round(values[0] * 255.0)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    with TemporaryDirectory(prefix="stegolab-model-proof-") as temporary:
        destination = Path(temporary) / "trial.png"
        with Image.fromarray(pixels) as image:
            write_png(image, destination)
        with Image.open(destination, formats=["PNG"]) as reopened:
            stored = np.array(reopened, dtype=np.uint8, copy=True)
    return (
        torch.from_numpy(stored).permute(2, 0, 1).contiguous().float().unsqueeze(0)
        / 255.0
    )


def _fixture_message(kind: FixtureKind, context: ProtocolContext) -> str:
    """Create public test content without accepting or persisting user secrets."""
    if kind == "empty":
        return ""
    if kind == "unicode":
        return PUBLIC_UNICODE_MESSAGE
    return "M" * calculate_capacity(context).maximum_message_bytes


def run_png_trial(
    encoder: nn.Module,
    decoder: nn.Module,
    cover: Tensor,
    source_identity: str,
    compatibility_identifier: str,
    fixture_kind: FixtureKind,
    deadline: float | None,
) -> PngEvaluationTrial:
    """Attempt one frame exactly once and measure integer-pixel recovery quality."""
    start = time.monotonic()
    height, width = cover.shape[-2:]
    context = ProtocolContext(
        width=width, height=height, compatibility_identifier=compatibility_identifier
    )
    error_code: str | None = None
    bit_error: float | None = None
    peak_ratio: float | None = None
    similarity: float | None = None
    clipped: float | None = None
    identical = recovered = False
    try:
        check_deadline(deadline)
        message = _fixture_message(fixture_kind, context)
        protected = encode_message(message, PUBLIC_FIXTURE_PASSWORD, context)
        payload = torch.from_numpy(create_payload_map(protected, context)).float()
        check_deadline(deadline)
        raw = encoder(cover, payload.unsqueeze(0))
        validate_model_tensor(raw, (1, 3, height, width))
        check_deadline(deadline)
        clipped = float(((raw < 0.0) | (raw > 1.0)).float().mean())
        reopened = _saved_pixels(quantize_image(raw))
        check_deadline(deadline)
        logits = decoder(reopened)
        validate_model_tensor(logits, (1, 1, height, width))
        check_deadline(deadline)
        bit_error = float(((logits[0] > 0) != payload.bool()).float().mean())
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
        error_code = "model_evaluation_failed"
    return PngEvaluationTrial(
        source_identity=source_identity,
        width=width,
        height=height,
        fixture_kind=fixture_kind,
        recovered=recovered,
        raw_bit_error_rate=bit_error,
        peak_signal_to_noise_ratio=peak_ratio,
        structural_similarity=similarity,
        identical_pixels=identical,
        clipped_fraction=clipped,
        elapsed_seconds=time.monotonic() - start,
        error_code=error_code,
    )
