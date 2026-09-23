"""Compare product behavior with independently generated static bytes."""

import hashlib

import pytest

from backend_service.message_correction import protect_frame
from backend_service.message_frame import (
    authenticated_context,
    create_frame,
    derive_key,
)
from backend_service.message_protocol import decode_message
from backend_service.payload_map import create_payload_map, recover_payload_bytes
from backend_service.protocol_fixtures import PublicProtocolVector, load_public_vectors


@pytest.mark.parametrize(
    "vector", load_public_vectors(), ids=lambda vector: vector.identifier
)
def test_independent_known_answers(
    vector: PublicProtocolVector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject public randomness only in tests; never regenerate expected bytes."""
    salt, nonce = bytes.fromhex(vector.salt_hex), bytes.fromhex(vector.nonce_hex)
    values = iter((salt, nonce))

    def public_randomness(size: int) -> bytes:
        """Use the labelled test values while checking each requested size."""
        value = next(values)
        assert len(value) == size
        return value

    monkeypatch.setattr("backend_service.message_frame.utils.random", public_randomness)
    context = vector.context()
    assert vector.message.encode().hex() == vector.message_utf8_hex
    assert derive_key(vector.password.encode(), salt).hex() == vector.expected_key_hex
    frame = create_frame(vector.message, vector.password, context)
    assert frame.hex() == vector.expected_frame_hex
    assert frame[:45].hex() == vector.expected_header_hex
    assert (
        authenticated_context(frame[:45], context).hex()
        == vector.expected_associated_data_hex
    )
    protected = protect_frame(frame, context)
    assert protected.hex() == vector.expected_protected_hex
    payload = create_payload_map(protected, context)
    assert (
        hashlib.sha256(payload.tobytes()).hexdigest()
        == vector.expected_payload_map_sha256
    )
    assert recover_payload_bytes(payload, context) == protected
    assert decode_message(protected, vector.password, context) == vector.message
