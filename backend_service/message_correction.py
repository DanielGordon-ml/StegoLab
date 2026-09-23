"""Explicit byte correction and interleaving for protocol version one."""

from reedsolo import ReedSolomonError, RSCodec  # type: ignore[import-untyped]

from backend_service.payload_capacity import calculate_capacity
from backend_service.protocol_failures import (
    invalid_protocol_input,
    protocol_resource_failure,
    recovery_failure,
)
from schemas.protocol import ProtocolContext


def correction_codec() -> RSCodec:
    """Construct the fixed GF(256) codec without relying on default parameters."""
    return RSCodec(nsym=32, nsize=255, fcr=0, prim=0x11D, generator=2, c_exp=8)


def protect_frame(frame: bytes, context: ProtocolContext) -> bytes:
    """Protect all header and ciphertext bytes, then interleave block columns."""
    capacity = calculate_capacity(context)
    if not isinstance(frame, bytes) or len(frame) != capacity.frame_bytes:
        raise invalid_protocol_input()
    try:
        codec = correction_codec()
        blocks = [
            bytes(codec.encode(frame[offset : offset + 223]))
            for offset in range(0, len(frame), 223)
        ]
        return bytes(block[column] for column in range(255) for block in blocks)
    except MemoryError:
        raise protocol_resource_failure() from None


def restore_frame(protected: bytes, context: ProtocolContext) -> bytes:
    """Undo column interleaving and correct blocks before authentication."""
    capacity = calculate_capacity(context)
    if (
        not isinstance(protected, bytes)
        or len(protected) * 8 != capacity.protected_bits
    ):
        raise recovery_failure()
    try:
        codec = correction_codec()
        return b"".join(
            bytes(codec.decode(protected[index :: capacity.block_count])[0])
            for index in range(capacity.block_count)
        )
    except (ReedSolomonError, ValueError, IndexError):
        raise recovery_failure() from None
    except MemoryError:
        raise protocol_resource_failure() from None
