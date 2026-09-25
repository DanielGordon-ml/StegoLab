"""Installed experimental model records and their explicit installation request."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, field_validator

from schemas.base import StrictRecord
from schemas.configuration import RequestIdentifier

ModelIdentifier = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"),
]
ExportReference = Annotated[str, Field(min_length=1, max_length=128)]


class InstalledModel(StrictRecord):
    """Describe one installed encoder/decoder pair without exposing its path."""

    model_identifier: ModelIdentifier
    compatibility_identifier: str = Field(pattern=r"^dense_v1_[0-9a-f]{64}$")
    source_identifier: str = Field(min_length=1, max_length=256)
    profile_identifier: Literal["test_only_v1"] = "test_only_v1"
    status: Literal["experimental"] = "experimental"
    format_version: Literal[1, 2]
    minimum_side: Literal[512] = 512
    maximum_side: Literal[1024] = 1024
    maximum_payload_bytes: Literal[1024] = 1024
    installed_at: AwareDatetime
    export_reference: ExportReference

    @field_validator("format_version", mode="before")
    @classmethod
    def require_integer_format(cls, value: object) -> object:
        """Reject booleans and decimal lookalikes for the package format version."""
        if type(value) is not int:
            raise ValueError("The package format version requires a whole number.")
        return value


class ModelInstallRequest(StrictRecord):
    """Identify one explicit installation so that network retries are safe."""

    client_request_identifier: RequestIdentifier
    export_reference: ExportReference
