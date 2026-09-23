"""Safe CLI demonstrations using only bundled public protocol examples."""

import hashlib
from collections.abc import Callable

from backend_service.failures import ApplicationFailure
from backend_service.message_correction import protect_frame
from backend_service.message_protocol import decode_message, encode_message
from backend_service.payload_capacity import calculate_capacity
from backend_service.payload_map import create_payload_map, recover_payload_bytes
from backend_service.protocol_fixtures import (
    PublicProtocolVector,
    load_public_vectors,
    verification_failure,
)
from schemas.protocol import ProtocolVerification


def require_match(condition: bool) -> None:
    """Stop a public verification run safely when expected data differs."""
    if not condition:
        raise verification_failure()


def require_failure(operation: Callable[[], object], code: str) -> None:
    """Check a documented rejection without printing its test inputs."""
    try:
        operation()
    except ApplicationFailure as failure:
        require_match(failure.code == code)
    else:
        raise verification_failure()


def verify_vector(vector: PublicProtocolVector) -> None:
    """Compare fixed frame, correction, map, and recovered text expectations."""
    context = vector.context()
    expected = bytes.fromhex(vector.expected_protected_hex)
    frame = bytes.fromhex(vector.expected_frame_hex)
    capacity = calculate_capacity(context)
    require_match(capacity.maximum_message_bytes == vector.candidate_message_bytes)
    require_match(capacity.block_count == vector.block_count)
    require_match(protect_frame(frame, context) == expected)
    payload = create_payload_map(expected, context)
    require_match(
        hashlib.sha256(payload.tobytes()).hexdigest()
        == vector.expected_payload_map_sha256
    )
    require_match(recover_payload_bytes(payload, context) == expected)
    require_match(decode_message(expected, vector.password, context) == vector.message)
    fresh = encode_message(vector.message, vector.password, context)
    require_match(decode_message(fresh, vector.password, context) == vector.message)


def verify_rejections(vector: PublicProtocolVector) -> None:
    """Demonstrate wrong credentials/context, correction, and capacity limits."""
    context = vector.context()
    protected = bytes.fromhex(vector.expected_protected_hex)
    capacity = calculate_capacity(context)
    require_failure(
        lambda: decode_message(protected, "wrong public password", context),
        "message_recovery_failed",
    )
    other_context = context.model_copy(
        update={"compatibility_identifier": "wrong_public_context"}
    )
    require_failure(
        lambda: decode_message(protected, vector.password, other_context),
        "message_recovery_failed",
    )
    require_failure(
        lambda: decode_message(bytes(len(protected)), vector.password, context),
        "message_recovery_failed",
    )
    damaged = bytearray(protected)
    for column in range(16):
        for block in range(capacity.block_count):
            damaged[column * capacity.block_count + block] ^= 0xB7
    require_match(
        decode_message(bytes(damaged), vector.password, context) == vector.message
    )
    require_failure(
        lambda: encode_message(
            "x" * (capacity.maximum_message_bytes + 1), vector.password, context
        ),
        "message_too_large",
    )


def verify_protocol() -> ProtocolVerification:
    """Run bundled public checks; accept no passwords or secret messages."""
    vectors = load_public_vectors()
    try:
        for vector in vectors:
            verify_vector(vector)
        verify_rejections(vectors[0])
    except (ValueError, ApplicationFailure):
        raise verification_failure() from None
    return ProtocolVerification(
        fixture_count=len(vectors),
        checks_passed=[
            "known_answers",
            "exact_utf8",
            "fresh_encryption",
            "wrong_password",
            "wrong_context",
            "correctable_damage",
            "invalid_payload",
            "capacity_overflow",
        ],
    )
