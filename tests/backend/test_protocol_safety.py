"""Authenticated malformed content, bounded work, and private failure checks."""

import logging
import struct
from pathlib import Path

import pytest
from nacl.secret import Aead

from backend_service.failures import ApplicationFailure
from backend_service.message_frame import (
    authenticated_context,
    create_frame,
    derive_key,
    recover_frame,
)
from backend_service.message_protocol import decode_message, encode_message
from backend_service.payload_capacity import calculate_capacity
from backend_service.protocol_failures import RECOVERY_MESSAGE
from schemas.protocol import ProtocolContext


@pytest.mark.parametrize("case", ["utf8", "padding", "length"])
def test_authenticated_malformed_body(case: str) -> None:
    """Authentication is necessary but invalid structure still returns no text."""
    context = ProtocolContext(width=512, height=512)
    password = "public password"
    frame = create_frame("", password, context)
    header = frame[:45]
    body = bytearray(calculate_capacity(context).frame_bytes - 61)
    if case == "utf8":
        body[:3] = b"\x00\x01\xff"
    elif case == "padding":
        body[-1] = 1
    else:
        body[:2] = struct.pack(">H", 257)
    key = derive_key(password.encode(), header[5:21])
    malformed = (
        header
        + Aead(key)
        .encrypt(bytes(body), authenticated_context(header, context), header[21:45])
        .ciphertext
    )
    with pytest.raises(ApplicationFailure, match=RECOVERY_MESSAGE):
        recover_frame(malformed, password, context)


def test_invalid_header_does_not_derive_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hostile header cannot start expensive work or select a cost profile."""

    def unexpected_work(*arguments: object, **keywords: object) -> bytes:
        """Flag costly processing of a plainly invalid frame."""
        pytest.fail("Invalid header reached key derivation.")

    monkeypatch.setattr("backend_service.message_frame.derive_key", unexpected_work)
    with pytest.raises(ApplicationFailure, match=RECOVERY_MESSAGE):
        recover_frame(bytes(446), "public", ProtocolContext(width=512, height=512))


def test_services_do_not_log_or_persist_secrets(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Successful and failed processing leave no message/password in output."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)
    secret = "PRIVATE_SENTINEL_83672"
    password = "PASSWORD_SENTINEL_72934"
    context = ProtocolContext(width=512, height=512)
    protected = encode_message(secret, password, context)
    assert decode_message(protected, password, context) == secret
    with pytest.raises(ApplicationFailure) as caught:
        decode_message(protected, "WRONG_PASSWORD_SENTINEL", context)
    captured = capsys.readouterr()
    output = caplog.text + captured.out + captured.err + str(caught.value)
    assert all(
        value not in output for value in (secret, password, "WRONG_PASSWORD_SENTINEL")
    )
    assert list(tmp_path.iterdir()) == []


def test_key_allocation_error_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A library resource exception cannot expose supplied values."""

    def failed_key(*arguments: object, **keywords: object) -> bytes:
        """Inject a private resource exception from the dependency."""
        raise MemoryError("PRIVATE_LIBRARY_SENTINEL")

    monkeypatch.setattr("backend_service.message_frame.pwhash.argon2id.kdf", failed_key)
    with pytest.raises(ApplicationFailure) as caught:
        create_frame("message", "password", ProtocolContext(width=512, height=512))
    assert caught.value.code == "protocol_resources_unavailable"
    assert "PRIVATE_LIBRARY_SENTINEL" not in str(caught.value)
