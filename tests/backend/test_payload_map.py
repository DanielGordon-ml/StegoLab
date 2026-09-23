"""Capacity and deterministic reconstruction checks without neural models."""

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from backend_service.failures import ApplicationFailure
from backend_service.message_protocol import decode_message, encode_message
from backend_service.payload_capacity import calculate_capacity
from backend_service.payload_map import create_payload_map, recover_payload_bytes
from schemas.protocol import ProtocolContext


@pytest.mark.parametrize(
    ("width", "height", "maximum", "blocks"),
    [
        (512, 512, 256, 2),
        (513, 517, 259, 2),
        (512, 766, 383, 2),
        (512, 768, 384, 3),
        (512, 1212, 606, 3),
        (512, 1214, 607, 4),
        (512, 1658, 829, 4),
        (512, 1660, 830, 5),
        (1024, 1024, 1024, 5),
        (3840, 2160, 1024, 5),
        (4096, 2160, 1024, 5),
    ],
)
def test_complete_layout(width: int, height: int, maximum: int, blocks: int) -> None:
    """All overhead and all repeated pixels are accounted for at each tier."""
    context = ProtocolContext(width=width, height=height)
    capacity = calculate_capacity(context)
    assert capacity.maximum_message_bytes == maximum
    assert capacity.block_count == blocks
    assert maximum + capacity.padding_bytes_at_capacity + 63 == blocks * 223
    assert capacity.frame_bytes + capacity.correction_bytes == blocks * 255
    assert capacity.protected_bits + capacity.repeated_bits == width * height
    protected = encode_message("a" * maximum, "public password", context)
    payload = create_payload_map(protected, context)
    assert payload.shape == (1, height, width)
    assert payload.dtype == np.uint8
    assert recover_payload_bytes(payload, context) == protected
    assert decode_message(protected, "public password", context) == "a" * maximum
    with pytest.raises(ApplicationFailure, match="byte limit"):
        encode_message("a" * (maximum + 1), "public password", context)


@pytest.mark.parametrize(
    ("width", "height"), [(511, 512), (4097, 512), (4096, 4096), (512, 0)]
)
def test_invalid_dimensions(width: int, height: int) -> None:
    """Reject unsupported sides and area before allocation."""
    with pytest.raises(ValidationError):
        ProtocolContext(width=width, height=height)


def test_revalidate_constructed_context() -> None:
    """Do not trust callers that bypass structural-model construction."""
    context = ProtocolContext.model_construct(width=4096, height=4096)
    with pytest.raises(ApplicationFailure):
        calculate_capacity(context)
    with pytest.raises(ValidationError):
        ProtocolContext.model_validate(
            {"width": 512, "height": 512, "protocol_version": True}
        )


def test_bit_order_and_repetition() -> None:
    """Known bits use MSB-first order, including the final partial repetition."""
    context = ProtocolContext(width=513, height=517)
    protected = b"\x81" + bytes(509)
    values = create_payload_map(protected, context).reshape(-1)
    assert list(values[:8]) == [1, 0, 0, 0, 0, 0, 0, 1]
    assert np.array_equal(values[:4080], values[4080:8160])
    assert values[-1] == values[(values.size - 1) % 4080]


def test_soft_recovery_and_ties() -> None:
    """Use logit strength, not hard threshold voting, and resolve ties to zero."""
    context = ProtocolContext(width=512, height=512)
    logits = np.zeros((1, 512, 512), dtype=np.float32)
    flat = logits.reshape(-1)
    flat[::4080] = -1
    flat[0] = 1000
    flat[1] = 1
    flat[4081] = -1
    restored = recover_payload_bytes(logits, context)
    assert restored == b"\x80" + bytes(509)
    bits = np.zeros_like(logits, dtype=np.uint8)
    bits.reshape(-1)[1024::4080][:32] = 1
    assert recover_payload_bytes(bits, context) == bytes(510)


@pytest.mark.parametrize("invalid", ["shape", "nan", "infinity", "dtype", "bits"])
def test_invalid_maps(invalid: str) -> None:
    """Reject malformed values without guessing or silently casting data."""
    context = ProtocolContext(width=512, height=512)
    values: NDArray[np.float32] | NDArray[np.float64] | NDArray[np.uint8]
    values = np.zeros((1, 512, 512), dtype=np.float32)
    if invalid == "shape":
        values = values.reshape(512, 512)
    elif invalid == "nan":
        values[0, 0, 0] = np.nan
    elif invalid == "infinity":
        values[0, 0, 0] = np.inf
    elif invalid == "dtype":
        values = values.astype(np.float64)
    else:
        values = values.astype(np.uint8)
        values[0, 0, 0] = 2
    with pytest.raises(ApplicationFailure):
        recover_payload_bytes(values, context)  # type: ignore[arg-type]
