"""Safe acceptance records for independent learned-package text recovery."""

from typing import Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from schemas.base import StrictRecord


class ExportRecoveryTrial(StrictRecord):
    """Describe one package round trip without messages, passwords, or file paths."""

    model_config = ConfigDict(allow_inf_nan=False)
    source_identity: str = Field(min_length=1)
    width: int = Field(ge=512, le=1024)
    height: int = Field(ge=512, le=1024)
    fixture_kind: Literal["empty", "unicode", "maximum"]
    recovered: bool
    elapsed_seconds: float = Field(ge=0.0)
    error_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,127}$")


class ExportRecoveryReport(StrictRecord):
    """Distinguish 36 exact package recoveries from incomplete or failed checks."""

    model_config = ConfigDict(allow_inf_nan=False)
    schema_version: Literal[1] = 1
    compatibility_identifier: str | None = Field(
        default=None, pattern=r"^dense_v1_[0-9a-f]{64}$"
    )
    source_identifier: str | None = None
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_trials: Literal[36] = 36
    trials: list[ExportRecoveryTrial] = Field(max_length=36)
    completed: bool
    exact_recovery_count: int = Field(ge=0, le=36)
    exact_recovery_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    recovery_gate_passed: bool
    stopped_reason: Literal["complete", "deadline", "failed"]
    elapsed_seconds: float = Field(ge=0.0)
    error_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,127}$")

    @field_validator("schema_version", "expected_trials", mode="before")
    @classmethod
    def exact_integer_literals(cls, value: object) -> object:
        """Reject boolean and float values that compare equal to fixed integers."""
        if type(value) is not int:
            raise ValueError(
                "Export recovery version and trial count must be integers."
            )
        return value

    @model_validator(mode="after")
    def consistent_results(self) -> Self:
        """Reject contradictory counts or claims of a complete recovery gate."""
        recovered = sum(trial.recovered for trial in self.trials)
        rate = recovered / len(self.trials) if self.trials else None
        complete = self.stopped_reason == "complete" and len(self.trials) == 36
        if (
            self.exact_recovery_count != recovered
            or self.exact_recovery_rate != rate
            or self.completed != complete
            or self.recovery_gate_passed != (complete and recovered == 36)
        ):
            raise ValueError("Export recovery counts and completion claims disagree.")
        return self
