"""Versioned requests and reports for bounded pilot preparation commands."""

from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from schemas.base import StrictRecord
from schemas.evaluation import PngEvaluationTrial


class PilotCriticConfiguration(StrictRecord):
    """Optional adversarial critic; off by default and never part of an export."""

    enabled: bool = False
    weight: float = Field(default=1.0, ge=0.0, le=100.0)
    learning_rate: float = Field(default=0.0001, ge=0.00001, le=0.001)
    weight_clip: float = Field(default=0.1, ge=0.001, le=1.0)
    hidden_channels: Literal[32] = 32

    @field_validator("enabled", mode="before")
    @classmethod
    def require_boolean(cls, value: object) -> object:
        """Reject numbers that would compare equal to a boolean."""
        if type(value) is not bool:
            raise ValueError("The critic switch must be true or false.")
        return value

    @field_validator("hidden_channels", mode="before")
    @classmethod
    def require_integer_channels(cls, value: object) -> object:
        """Prevent booleans and decimal lookalikes from matching the fixed width."""
        if type(value) is not int:
            raise ValueError("The critic width requires a whole number.")
        return value


class PilotTrainingConfiguration(StrictRecord):
    """Freeze the numerical profile before the first optimizer update."""

    architecture: Literal["dense_pilot_v1"] = "dense_pilot_v1"
    device: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["float32"] = "float32"
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    crop_size: Literal[256] = 256
    physical_batch_size: Literal[4, 8] = 4
    effective_batch_size: Literal[16] = 16
    learning_rate: float = Field(default=0.0001, ge=0.0001, le=0.0001)
    planned_optimizer_steps: int = Field(ge=1, le=100_000_000)
    maximum_image_loss_weight: float = Field(default=100.0, ge=100.0, le=100.0)
    checkpoint_interval_seconds: Literal[300] = 300
    validation_interval_steps: int = Field(default=100, ge=1, le=100_000)
    cpu_threads: int = Field(default=4, ge=1, le=16)
    critic: PilotCriticConfiguration = Field(default_factory=PilotCriticConfiguration)

    @field_validator(
        "crop_size",
        "physical_batch_size",
        "effective_batch_size",
        "checkpoint_interval_seconds",
        mode="before",
    )
    @classmethod
    def require_integer_literals(cls, value: object) -> object:
        """Prevent booleans and decimal lookalikes from matching fixed integers."""
        if type(value) is not int:
            raise ValueError("This setting requires a whole number.")
        return value

    @property
    def image_loss_ramp_steps(self) -> int:
        """Resolve the fixed twenty-percent schedule without rounding ambiguity."""
        return (self.planned_optimizer_steps + 4) // 5


class PilotRequest(StrictRecord):
    """Identify a CPU smoke invocation or an explicitly guarded GPU session."""

    schema_version: Literal[2] = 2
    experiment_identifier: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    output_root: str = Field(default=".", min_length=1, max_length=4096)
    execution_mode: Literal["cpu_smoke", "gpu_pilot"] = "cpu_smoke"
    session_identifier: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"
    )
    cpu_smoke_timeout_seconds: float = Field(default=120.0, ge=1.0, le=120.0)

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_integer_version(cls, value: object) -> object:
        """Reject permissive comparisons at the protocol version boundary."""
        if type(value) is not int:
            raise ValueError("The schema version requires a whole number.")
        return value

    @model_validator(mode="after")
    def validate_session(self) -> "PilotRequest":
        """Require a host session for GPU work and keep smoke work separate."""
        if (self.execution_mode == "gpu_pilot") != (
            self.session_identifier is not None
        ):
            raise ValueError("Only GPU pilot commands require a session identifier.")
        return self


class PilotTrainingRequest(PilotRequest):
    """Start or exactly resume a frozen profile without extending smoke limits."""

    dataset_directory: str = Field(min_length=1, max_length=4096)
    configuration: PilotTrainingConfiguration
    resume_checkpoint: str | None = Field(default=None, min_length=1, max_length=4096)
    fork_from_checkpoint: str | None = Field(
        default=None, min_length=1, max_length=4096
    )
    stop_after_step: int | None = Field(default=None, ge=1, le=100_000_000)

    @model_validator(mode="after")
    def validate_execution(self) -> "PilotTrainingRequest":
        """Keep smoke training below ten absolute updates and prohibit fallback."""
        if self.resume_checkpoint is not None and self.fork_from_checkpoint is not None:
            raise ValueError("Choose either resume or fork, not both.")
        target = self.stop_after_step or self.configuration.planned_optimizer_steps
        if target > self.configuration.planned_optimizer_steps:
            raise ValueError("The stop step exceeds the frozen training schedule.")
        expected = "cpu" if self.execution_mode == "cpu_smoke" else "cuda"
        if self.configuration.device != expected:
            raise ValueError("The requested device does not match the execution mode.")
        if self.execution_mode == "cpu_smoke" and target > 10:
            raise ValueError("CPU smoke work is limited to ten optimizer updates.")
        return self


class PilotEvaluationRequest(PilotRequest):
    """Measure tuning cases on GPU or run one unranked CPU protocol smoke case."""

    checkpoint: str = Field(min_length=1, max_length=4096)
    dataset_directory: str = Field(min_length=1, max_length=4096)
    device: Literal["cpu", "cuda"] = "cpu"

    @model_validator(mode="after")
    def validate_device(self) -> "PilotEvaluationRequest":
        """Require the explicitly selected execution device."""
        expected = "cpu" if self.execution_mode == "cpu_smoke" else "cuda"
        if self.device != expected:
            raise ValueError("The requested device does not match the execution mode.")
        return self


class PilotExportRequest(PilotRequest):
    """Publish independent packages from a verified pilot checkpoint."""

    checkpoint: str = Field(min_length=1, max_length=4096)
    export_name: str = Field(
        default="002V", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
    )
    device: Literal["cpu", "cuda"] = "cpu"

    @model_validator(mode="after")
    def validate_runtime_device(self) -> "PilotExportRequest":
        """Require a guarded GPU session for any requested CUDA runtime checks."""
        if self.device == "cuda" and self.execution_mode != "gpu_pilot":
            raise ValueError("CUDA exports require an explicit GPU pilot session.")
        return self


class PilotTrainingRun(StrictRecord):
    """Report completed updates without implying model quality or GPU acceptance."""

    schema_version: Literal[2] = 2
    experiment_identifier: str
    execution_mode: Literal["cpu_smoke", "gpu_pilot"]
    status: Literal["completed", "stopped", "budget_exhausted"]
    global_step: int = Field(ge=0)
    checkpoint: str
    dataset_revision: str
    compatibility_identifier: str
    elapsed_seconds: float = Field(ge=0)
    quality_selection_performed: bool = False
    learning_gate: Literal["not_measured"] = "not_measured"
    stop_signal: Literal[2, 15] | None = None


class PilotTrainingStep(StrictRecord):
    """Keep finite scalar training diagnostics separate from recovery acceptance."""

    model_config = ConfigDict(allow_inf_nan=False)
    global_step: int = Field(ge=1)
    bit_loss: float = Field(ge=0)
    image_loss: float = Field(ge=0)
    image_loss_weight: float = Field(ge=0, le=100)
    critic_loss: float | None = None
    elapsed_seconds: float = Field(ge=0)


class PilotEvaluationReport(StrictRecord):
    """Retain every attempted case without claiming final benchmark acceptance."""

    model_config = ConfigDict(allow_inf_nan=False)
    schema_version: Literal[2] = 2
    kind: Literal["cpu_smoke", "tuning"]
    compatibility_identifier: str
    dataset_revision: str
    selection_identifier: str
    completed: bool
    expected_png_trials: int = Field(ge=1, le=32)
    png_trials: list[PngEvaluationTrial]
    exact_recovery_count: int = Field(ge=0, le=32)
    exact_recovery_rate: float | None = Field(default=None, ge=0, le=1)
    median_peak_signal_to_noise_ratio: float | None = None
    median_structural_similarity: float | None = Field(default=None, ge=-1, le=1)
    mean_clipped_fraction: float | None = Field(default=None, ge=0, le=1)
    elapsed_seconds: float = Field(ge=0)
    peak_process_memory_bytes: int = Field(ge=0)
    peak_device_memory_bytes: int = Field(ge=0)
    quality_measured: bool
    release_qualified: Literal[False] = False
    stopped_reason: Literal["complete", "deadline", "failed"]
