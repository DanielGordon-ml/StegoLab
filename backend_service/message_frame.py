"""Versioned authenticated frames; secrets are never structural records."""

import struct

from nacl import exceptions, pwhash, utils
from nacl.secret import Aead

from backend_service.failures import ApplicationFailure
from backend_service.payload_capacity import calculate_capacity, validate_context
from backend_service.protocol_failures import (
    protocol_resource_failure,
    recovery_failure,
)
from schemas.protocol import ProtocolContext

PROTOCOL_PREFIX = b"SGLB\x01"
AUTHENTICATION_DOMAIN = b"StegoLab/message_protocol/v1\x00"
HEADER_BYTES = 45
SALT_BYTES = 16
NONCE_BYTES = 24
KEY_BYTES = 32
KEY_MEMORY_BYTES = 64 * 1024 * 1024
KEY_OPERATIONS = 3


def password_bytes(password: str) -> bytes:
    """Preserve valid passwords exactly and enforce the project input limit."""
    try:
        if not isinstance(password, str) or not 1 <= len(password) <= 1024:
            raise ValueError
        encoded = password.encode("utf-8", errors="strict")
        if len(encoded) > 1024:
            raise ValueError
        return encoded
    except (ValueError, UnicodeError):
        raise ApplicationFailure(
            "invalid_password",
            "Enter a password containing 1 to 1,024 UTF-8 bytes.",
            422,
        ) from None


def authenticated_context(header: bytes, context: ProtocolContext) -> bytes:
    """Encode trusted context unambiguously with fixed-width dimensions."""
    identity = context.compatibility_identifier.encode("ascii")
    return (
        AUTHENTICATION_DOMAIN
        + header
        + struct.pack(">H", len(identity))
        + identity
        + struct.pack(">II", context.width, context.height)
    )


def derive_key(password: bytes, salt: bytes) -> bytes:
    """Use fixed Argon2id parameters, independent of incoming frame bytes."""
    try:
        return pwhash.argon2id.kdf(
            KEY_BYTES,
            password,
            salt,
            opslimit=KEY_OPERATIONS,
            memlimit=KEY_MEMORY_BYTES,
        )
    except (exceptions.RuntimeError, MemoryError):
        raise protocol_resource_failure() from None


def create_frame(message: str, password: str, context: ProtocolContext) -> bytes:
    """Pad and encrypt exact text with independently generated salt and nonce."""
    context = validate_context(context)
    capacity = calculate_capacity(context)
    try:
        if not isinstance(message, str):
            raise UnicodeError
        if len(message) > capacity.maximum_message_bytes:
            raise OverflowError
        encoded = message.encode("utf-8", errors="strict")
        if len(encoded) > capacity.maximum_message_bytes:
            raise OverflowError
    except OverflowError:
        raise ApplicationFailure(
            "message_too_large",
            "The message exceeds this image profile's UTF-8 byte limit.",
            422,
        ) from None
    except UnicodeError:
        raise ApplicationFailure(
            "invalid_message", "The message must contain valid UTF-8 text.", 422
        ) from None
    password_data = password_bytes(password)
    try:
        salt = utils.random(SALT_BYTES)
        nonce = utils.random(NONCE_BYTES)
        header = PROTOCOL_PREFIX + salt + nonce
        body = struct.pack(">H", len(encoded)) + encoded
        body = body.ljust(capacity.frame_bytes - HEADER_BYTES - Aead.MACBYTES, b"\x00")
        key = derive_key(password_data, salt)
        return (
            header
            + Aead(key)
            .encrypt(body, authenticated_context(header, context), nonce)
            .ciphertext
        )
    except (exceptions.CryptoError, MemoryError, OSError):
        raise protocol_resource_failure() from None


def recover_frame(frame: bytes, password: str, context: ProtocolContext) -> str:
    """Return text only after authentication, length, padding, and UTF-8 checks."""
    context = validate_context(context)
    capacity = calculate_capacity(context)
    password_data = password_bytes(password)
    if (
        not isinstance(frame, bytes)
        or len(frame) != capacity.frame_bytes
        or frame[:5] != PROTOCOL_PREFIX
    ):
        raise recovery_failure()
    header = frame[:HEADER_BYTES]
    salt, nonce = header[5:21], header[21:45]
    key = derive_key(password_data, salt)
    try:
        body = Aead(key).decrypt(
            frame[HEADER_BYTES:], authenticated_context(header, context), nonce
        )
        message_length = int.from_bytes(body[:2], "big")
        if message_length > capacity.maximum_message_bytes:
            raise ValueError
        if any(body[2 + message_length :]):
            raise ValueError
        return body[2 : 2 + message_length].decode("utf-8", errors="strict")
    except (exceptions.CryptoError, ValueError, UnicodeError):
        raise recovery_failure() from None
    except MemoryError:
        raise protocol_resource_failure() from None
