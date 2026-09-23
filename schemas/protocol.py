"""Safe structural records for the experimental message protocol."""

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from schemas.base import StrictRecord
from schemas.image_dimensions import ImageDimensions


class ProtocolContext(ImageDimensions):
    """Bind a trusted caller's dimensions and compatibility identity."""

    model_config = ConfigDict(frozen=True)
    protocol_version: Literal[1] = 1
    profile_identifier: Literal["test_only_v1"] = "test_only_v1"
    compatibility_identifier: str = Field(
        default="public_fixture_v1", pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$"
    )

    @field_validator("protocol_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        """Reject booleans and floats that compare equal to version one."""
        if type(value) is not int:
            raise ValueError("The protocol version must be an integer.")
        return value


class PayloadCapacity(StrictRecord):
    """Distinguish layout capacity from an available trained model."""

    profile_identifier: Literal["test_only_v1"] = "test_only_v1"
    maximum_message_bytes: int = Field(ge=256, le=1024)
    header_bytes: Literal[45] = 45
    message_length_bytes: Literal[2] = 2
    authentication_bytes: Literal[16] = 16
    padding_bytes_at_capacity: int = Field(ge=0, lt=223)
    block_count: int = Field(ge=2, le=5)
    frame_bytes: int = Field(ge=446, le=1115)
    correction_bytes: int = Field(ge=64, le=160)
    protected_bits: int = Field(ge=4080, le=10200)
    payload_map_bits: int = Field(ge=262144, le=8_850_000)
    repeated_bits: int = Field(ge=0)
    minimum_repetitions: int = Field(ge=1)
    additional_repetitions: int = Field(ge=0)


class ProtocolVerification(StrictRecord):
    """Report only public fixture checks, never messages or passwords."""

    profile_identifier: Literal["test_only_v1"] = "test_only_v1"
    fixture_count: int = Field(ge=1)
    checks_passed: list[str] = Field(min_length=1)
    status: Literal["passed"] = "passed"
