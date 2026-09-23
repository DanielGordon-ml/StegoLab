"""One-channel test maps with deterministic repetition and reconstruction."""

import numpy as np
from numpy.typing import NDArray

from backend_service.payload_capacity import calculate_capacity
from backend_service.protocol_failures import (
    invalid_protocol_input,
    protocol_resource_failure,
)
from schemas.protocol import ProtocolContext


def create_payload_map(protected: bytes, context: ProtocolContext) -> NDArray[np.uint8]:
    """Repeat MSB-first protected bits over every position in row order."""
    capacity = calculate_capacity(context)
    if (
        not isinstance(protected, bytes)
        or len(protected) * 8 != capacity.protected_bits
    ):
        raise invalid_protocol_input()
    try:
        bits = np.unpackbits(np.frombuffer(protected, dtype=np.uint8), bitorder="big")
        return np.resize(bits, capacity.payload_map_bits).reshape(
            1, context.height, context.width
        )
    except MemoryError:
        raise protocol_resource_failure() from None


def recover_payload_bytes(
    values: NDArray[np.uint8] | NDArray[np.float32], context: ProtocolContext
) -> bytes:
    """Vote hard bits or sum finite float32 logits; exact ties recover as zero."""
    capacity = calculate_capacity(context)
    if not isinstance(values, np.ndarray) or values.shape != (
        1,
        context.height,
        context.width,
    ):
        raise invalid_protocol_input()
    if values.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
        raise invalid_protocol_input()
    try:
        is_bits = values.dtype == np.dtype(np.uint8)
        if is_bits and np.any(values > 1):
            raise invalid_protocol_input()
        if not is_bits and not np.all(np.isfinite(values)):
            raise invalid_protocol_input()
        flat = values.reshape(-1)
        full_count, extra = divmod(flat.size, capacity.protected_bits)
        full_length = full_count * capacity.protected_bits
        sums = flat[:full_length].reshape(full_count, -1).sum(axis=0, dtype=np.float64)
        sums[:extra] += flat[full_length:]
        if is_bits:
            sums *= 2
            sums -= full_count
            sums[:extra] -= 1
        return np.packbits(sums > 0, bitorder="big").tobytes()
    except MemoryError:
        raise protocol_resource_failure() from None
