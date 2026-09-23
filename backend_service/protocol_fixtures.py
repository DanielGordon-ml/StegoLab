"""Load labelled public compatibility vectors bundled with the application."""

from importlib.resources import files
from typing import Literal

from pydantic import Field, ValidationError

from backend_service.failures import ApplicationFailure
from schemas.base import StrictRecord
from schemas.protocol import ProtocolContext


class PublicProtocolVector(StrictRecord):
    """Describe public examples, never submitted messages or credentials."""

    identifier: str
    width: int
    height: int
    compatibility_identifier: str
    message: str = Field(repr=False)
    password: str = Field(repr=False)
    salt_hex: str
    nonce_hex: str
    candidate_message_bytes: int
    block_count: int
    message_utf8_hex: str
    expected_key_hex: str
    expected_header_hex: str
    expected_associated_data_hex: str
    expected_frame_hex: str
    expected_protected_hex: str
    expected_payload_map_sha256: str

    def context(self) -> ProtocolContext:
        """Build trusted context from this bundled compatibility example."""
        return ProtocolContext(
            width=self.width,
            height=self.height,
            compatibility_identifier=self.compatibility_identifier,
        )


class PublicProtocolFixtures(StrictRecord):
    """Validate the structure and explicit public-only warning of the bundle."""

    protocol_version: Literal[1]
    warning: Literal[
        "PUBLIC TESTS ONLY: never reuse these passwords, salts, or nonces."
    ]
    derivation: dict[str, str | int]
    vectors: list[PublicProtocolVector] = Field(min_length=1)


def verification_failure() -> ApplicationFailure:
    """Report a fixture mismatch without displaying intermediate contents."""
    return ApplicationFailure(
        "protocol_verification_failed",
        "The public protocol checks failed. Reinstall the matching application "
        "version and retry.",
    )


def load_public_vectors() -> list[PublicProtocolVector]:
    """Load packaged public vectors without consulting a user-supplied path."""
    try:
        content = files("backend_service").joinpath("fixtures/protocol_v1.json")
        return PublicProtocolFixtures.model_validate_json(
            content.read_text(encoding="utf-8")
        ).vectors
    except (OSError, UnicodeError, ValidationError):
        raise verification_failure() from None
