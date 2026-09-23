"""Account for every byte and repeated bit in the test-only layout."""

from pydantic import ValidationError

from backend_service.protocol_failures import invalid_protocol_input
from schemas.protocol import PayloadCapacity, ProtocolContext


def validate_context(context: ProtocolContext) -> ProtocolContext:
    """Revalidate even constructed records before using sizes or identities."""
    try:
        return ProtocolContext.model_validate(context)
    except ValidationError:
        raise invalid_protocol_input() from None


def calculate_capacity(context: ProtocolContext) -> PayloadCapacity:
    """Return internal layout accounting, without advertising model support."""
    context = validate_context(context)
    pixels = context.width * context.height
    maximum = min(1024, pixels // 1024)
    blocks = (maximum + 63 + 222) // 223
    protected_bits = blocks * 255 * 8
    if protected_bits > pixels:
        raise invalid_protocol_input()
    return PayloadCapacity(
        maximum_message_bytes=maximum,
        padding_bytes_at_capacity=blocks * 223 - 63 - maximum,
        block_count=blocks,
        frame_bytes=blocks * 223,
        correction_bytes=blocks * 32,
        protected_bits=protected_bits,
        payload_map_bits=pixels,
        repeated_bits=pixels - protected_bits,
        minimum_repetitions=pixels // protected_bits,
        additional_repetitions=pixels % protected_bits,
    )
