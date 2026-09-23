"""Common validation behavior for application records."""

from pydantic import BaseModel, ConfigDict


class StrictRecord(BaseModel):
    """Reject unexpected fields, automatic conversions, and invalid defaults."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        json_schema_serialization_defaults_required=True,
    )
