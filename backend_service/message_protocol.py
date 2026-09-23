"""Reusable message services, independent of models, images, and storage."""

from backend_service.message_correction import protect_frame, restore_frame
from backend_service.message_frame import create_frame, recover_frame
from schemas.protocol import ProtocolContext


def encode_message(message: str, password: str, context: ProtocolContext) -> bytes:
    """Return protected frame bytes for a controlled test channel."""
    return protect_frame(create_frame(message, password, context), context)


def decode_message(protected: bytes, password: str, context: ProtocolContext) -> str:
    """Correct and authenticate a frame before returning the exact message."""
    return recover_frame(restore_frame(protected, context), password, context)
