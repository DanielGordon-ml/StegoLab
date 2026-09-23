"""Settings and mutation request records shared by all clients."""

from typing import Annotated, Literal

from pydantic import Field, field_validator

from schemas.base import StrictRecord

RequestIdentifier = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
]


class ConfigurationProfile(StrictRecord):
    """Store the checkpoint frequency as a whole number of minutes in seconds."""

    schema_version: Literal[1] = 1
    checkpoint_interval_seconds: Annotated[int, Field(gt=0, multiple_of=60)] = 300

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        """Reject boolean and floating-point values for the schema version."""
        if type(value) is not int:
            raise ValueError("The schema version must be a whole number.")
        return value


class ConfigurationUpdate(StrictRecord):
    """Identify a settings change so that network retries are safe."""

    client_request_identifier: RequestIdentifier
    configuration: ConfigurationProfile


class ConfigurationReset(StrictRecord):
    """Identify a request to restore and save the default settings."""

    client_request_identifier: RequestIdentifier
