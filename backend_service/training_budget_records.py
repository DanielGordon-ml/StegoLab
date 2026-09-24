"""Strict on-disk accounting for the bounded local CPU proof."""

from typing import Literal

from pydantic import Field, model_validator

from schemas.base import StrictRecord


class ExperimentBudget(StrictRecord):
    """Count every invocation belonging to one fixed experiment identity."""

    experiment_identifier: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    consumed_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class ActiveBudget(StrictRecord):
    """Reserve the remaining session allowance before any proof work can start."""

    experiment_identifier: str
    started_at: float = Field(ge=0, allow_inf_nan=False)
    reserved_seconds: float = Field(gt=0, le=7200, allow_inf_nan=False)


class BudgetLedger(StrictRecord):
    """Keep spent time, experiment slots, and crash reservations across restarts."""

    schema_version: Literal[1] = 1
    maximum_experiments: Literal[2] = 2
    maximum_total_seconds: Literal[14400] = 14400
    maximum_experiment_seconds: Literal[7200] = 7200
    save_reserve_seconds: Literal[1200] = 1200
    consumed_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    experiments: list[ExperimentBudget] = Field(default_factory=list, max_length=2)
    active: ActiveBudget | None = None

    @model_validator(mode="after")
    def consistent_accounting(self) -> "BudgetLedger":
        """Reject forged totals, duplicate experiments, and unrelated reservations."""
        names = [item.experiment_identifier for item in self.experiments]
        if len(names) != len(set(names)):
            raise ValueError("Experiment identifiers must be unique.")
        total = sum(item.consumed_seconds for item in self.experiments)
        if abs(total - self.consumed_seconds) > 1e-6:
            raise ValueError("Experiment and project time accounting disagree.")
        if self.active is not None and self.active.experiment_identifier not in names:
            raise ValueError("The active reservation has no experiment.")
        return self
