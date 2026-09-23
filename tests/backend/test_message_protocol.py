"""Exact recovery, authentication, and error-correction acceptance cases."""

import pytest

from backend_service.failures import ApplicationFailure
from backend_service.message_correction import protect_frame, restore_frame
from backend_service.message_frame import create_frame, recover_frame
from backend_service.message_protocol import decode_message, encode_message
from backend_service.payload_capacity import calculate_capacity
from backend_service.protocol_failures import RECOVERY_MESSAGE
from schemas.protocol import ProtocolContext

PUBLIC_PASSWORD = "public test password שלום 🔐"
CONTEXT = ProtocolContext(width=512, height=512)


@pytest.mark.parametrize(
    "message", ["", "  \n\t ", "שלום 🌍 e\u0301\ntext\r\n", "a" * 256, "🙂" * 64]
)
def test_exact_recovery(message: str) -> None:
    """Retain original UTF-8 bytes, including empty and boundary messages."""
    protected = encode_message(message, PUBLIC_PASSWORD, CONTEXT)
    assert decode_message(protected, PUBLIC_PASSWORD, CONTEXT) == message


def test_fresh_randomness_and_fixed_length() -> None:
    """Each frame has independent salt/nonce and hides actual message length."""
    frames = [create_frame(message, PUBLIC_PASSWORD, CONTEXT) for message in ("", "a")]
    assert frames[0][5:21] != frames[1][5:21]
    assert frames[0][21:45] != frames[1][21:45]
    assert len(frames[0]) == len(frames[1]) == 446


@pytest.mark.parametrize("message", ["a" * 257, "🙂" * 65])
def test_capacity_overflow(message: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject UTF-8 overflow before spending resources on key derivation."""

    def unexpected_work(*arguments: object, **keywords: object) -> bytes:
        """Fail if rejected input reaches expensive work."""
        pytest.fail("Key derivation must not run for overflow.")

    monkeypatch.setattr("backend_service.message_frame.derive_key", unexpected_work)
    with pytest.raises(ApplicationFailure, match="UTF-8 byte limit"):
        encode_message(message, PUBLIC_PASSWORD, CONTEXT)


@pytest.mark.parametrize("password", ["", "a" * 1025, "🙂" * 257, "\ud800"])
def test_invalid_password(password: str) -> None:
    """Reject invalid password inputs through safe fixed errors."""
    with pytest.raises(ApplicationFailure, match="1 to 1,024"):
        create_frame("public", password, CONTEXT)


def test_password_bytes_are_not_normalized() -> None:
    """Spaces, combining characters, and zero bytes are significant."""
    password = " e\u0301\x00 "
    protected = encode_message("public", password, CONTEXT)
    assert decode_message(protected, password, CONTEXT) == "public"
    for changed in (password.strip(), " é\x00 "):
        with pytest.raises(ApplicationFailure, match=RECOVERY_MESSAGE):
            decode_message(protected, changed, CONTEXT)


def test_wrong_password_and_context_have_same_error() -> None:
    """Never reveal partial text after a password or identity mismatch."""
    protected = encode_message("PUBLIC_SECRET_SENTINEL", PUBLIC_PASSWORD, CONTEXT)
    contexts = [
        CONTEXT,
        ProtocolContext(width=512, height=512, compatibility_identifier="other_model"),
        ProtocolContext(width=512, height=513),
    ]
    failures = []
    for context in contexts:
        password = "wrong" if context is CONTEXT else PUBLIC_PASSWORD
        with pytest.raises(ApplicationFailure) as caught:
            decode_message(protected, password, context)
        failures.append((caught.value.code, str(caught.value)))
    assert failures == [("message_recovery_failed", RECOVERY_MESSAGE)] * 3


@pytest.mark.parametrize("damaged_symbols", [1, 8, 16, 17, 33])
def test_symbol_correction(damaged_symbols: int) -> None:
    """Count byte errors per codeword, including header and parity bytes."""
    protected = encode_message("exact message", PUBLIC_PASSWORD, CONTEXT)
    capacity = calculate_capacity(CONTEXT)
    changed = bytearray(protected)
    for block in range(capacity.block_count):
        for symbol in range(damaged_symbols):
            position = ((symbol * 13) % 255) * capacity.block_count + block
            changed[position] ^= 0xB7
    try:
        recovered = decode_message(bytes(changed), PUBLIC_PASSWORD, CONTEXT)
        assert recovered == "exact message"
    except ApplicationFailure as failure:
        assert damaged_symbols > 16
        assert str(failure) == RECOVERY_MESSAGE


def test_sixteen_errors_across_header_data_and_parity() -> None:
    """Exercise the full correction promise in every five-block codeword."""
    context = ProtocolContext(width=1024, height=1024)
    message = "x" * 1024
    protected = encode_message(message, PUBLIC_PASSWORD, context)
    changed = bytearray(protected)
    positions = [0, 4, 5, 20, 21, 44, 45, 100, 200, 222, 223, 224, 230, 240, 253, 254]
    for block in range(5):
        for position in positions:
            changed[position * 5 + block] ^= 0xA5
    assert decode_message(bytes(changed), PUBLIC_PASSWORD, context) == message


@pytest.mark.parametrize("damage", ["nonce", "ciphertext", "version", "length"])
def test_authenticated_failure_after_valid_correction(damage: str) -> None:
    """A valid correction code cannot make an altered encrypted frame valid."""
    frame = bytearray(create_frame("public", PUBLIC_PASSWORD, CONTEXT))
    if damage == "length":
        protected = b"\x00" * 13
    else:
        position = {"nonce": 21, "ciphertext": 70, "version": 4}[damage]
        frame[position] ^= 1
        protected = protect_frame(bytes(frame), CONTEXT)
    with pytest.raises(ApplicationFailure, match=RECOVERY_MESSAGE):
        decode_message(protected, PUBLIC_PASSWORD, CONTEXT)


def test_invalid_utf8_input() -> None:
    """Do not replace invalid Unicode characters silently."""
    with pytest.raises(ApplicationFailure, match="valid UTF-8 text"):
        create_frame("prefix\ud800suffix", PUBLIC_PASSWORD, CONTEXT)


def test_zero_payload_and_roundtrip_layers() -> None:
    """A blank map does not yield a message; layers preserve valid frame bytes."""
    frame = create_frame("public", PUBLIC_PASSWORD, CONTEXT)
    assert restore_frame(protect_frame(frame, CONTEXT), CONTEXT) == frame
    assert recover_frame(frame, PUBLIC_PASSWORD, CONTEXT) == "public"
    with pytest.raises(ApplicationFailure, match=RECOVERY_MESSAGE):
        decode_message(bytes(510), PUBLIC_PASSWORD, CONTEXT)
