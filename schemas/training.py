"""Strict requests and safe summaries for the experimental CPU workflow."""

from typing import Literal

from pydantic import Field, field_validator

from schemas.base import StrictRecord


class TrainingConfiguration(StrictRecord):
    """Freeze the agreed development profile independently of a stop boundary."""

    architecture: Literal["dense_cpu_v1"] = "dense_cpu_v1"
    seed: Literal[0] = 0
    device: Literal["cpu"] = "cpu"
    crop_size: Literal[256] = 256
    physical_batch_size: Literal[1] = 1
    effective_batch_size: Literal[4] = 4
    learning_rate: float = Field(default=0.0001, ge=0.0001, le=0.0001)
    planned_optimizer_steps: Literal[1000] = 1000
    image_loss_ramp_steps: Literal[200] = 200
    maximum_image_loss_weight: float = Field(default=100.0, ge=100.0, le=100.0)
    checkpoint_interval_seconds: Literal[300] = 300
    validation_interval_steps: Literal[100] = 100
    cpu_threads: int = Field(default=4, ge=1, le=16)

    @field_validator("*", mode="before")
    @classmethod
    def reject_boolean_settings(cls, value: object) -> object:
        """Prevent booleans from satisfying integer-valued fixed settings."""
        if isinstance(value, bool):
            raise ValueError("Training settings cannot be boolean values.")
        return value

    @field_validator(
        "seed",
        "crop_size",
        "physical_batch_size",
        "effective_batch_size",
        "planned_optimizer_steps",
        "image_loss_ramp_steps",
        "checkpoint_interval_seconds",
        "validation_interval_steps",
        mode="before",
    )
    @classmethod
    def require_integer_settings(cls, value: object) -> object:
        """Keep floating-point lookalikes out of fixed integer settings."""
        if type(value) is not int:
            raise ValueError("This training setting requires a whole number.")
        return value


class ProofRequest(StrictRecord):
    """Locate one persistent experiment without exposing paths in reports."""

    schema_version: Literal[1] = 1
    experiment_identifier: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    output_root: str = Field(default=".", min_length=1, max_length=4096)

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_integer_version(cls, value: object) -> object:
        """Reject booleans and floats that compare equal to schema version one."""
        if type(value) is not int:
            raise ValueError("The schema version requires a whole number.")
        return value


class TrainingRequest(ProofRequest):
    """Start or explicitly resume the fixed CPU training profile."""

    dataset_directory: str = Field(min_length=1, max_length=4096)
    configuration: TrainingConfiguration = Field(default_factory=TrainingConfiguration)
    resume_checkpoint: str | None = Field(default=None, min_length=1, max_length=4096)
    stop_after_step: int = Field(default=1000, ge=1, le=1000)


class EvaluationRequest(ProofRequest):
    """Evaluate frozen checkpoint weights on the selected tuning cases."""

    checkpoint: str = Field(min_length=1, max_length=4096)
    dataset_directory: str = Field(min_length=1, max_length=4096)
    lightweight: bool = False


class ExportRequest(ProofRequest):
    """Create two independent packages from a verified paired checkpoint."""

    checkpoint: str = Field(min_length=1, max_length=4096)
    export_name: str = Field(
        default="001V", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
    )


class TrainingRun(StrictRecord):
    """Report training work separately from unmeasured learning gates."""

    schema_version: Literal[1] = 1
    experiment_identifier: str
    status: Literal["completed", "stopped", "budget_exhausted"]
    global_step: int = Field(ge=0, le=1000)
    checkpoint: str
    dataset_revision: str
    compatibility_identifier: str
    elapsed_seconds: float = Field(ge=0)
    remaining_experiment_seconds: float = Field(ge=0)
    selected_training_identities: list[str]
    selected_tuning_identities: list[str]
    learning_gate: Literal["not_measured"] = "not_measured"
    stop_signal: Literal[2, 15] | None = None


class TrainingStep(StrictRecord):
    """Store scalar progress without images, messages, or passwords."""

    global_step: int = Field(ge=1, le=1000)
    bit_loss: float = Field(ge=0, allow_inf_nan=False)
    image_loss: float = Field(ge=0, allow_inf_nan=False)
    image_loss_weight: float = Field(ge=0, le=100, allow_inf_nan=False)
    elapsed_seconds: float = Field(ge=0)
