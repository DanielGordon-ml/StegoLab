"""Strict instance-session accounting, separate from the local CPU proof."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from schemas.base import StrictRecord

PilotStage = Literal["setup", "baseline", "ablation", "evaluation"]
STAGE_SECONDS: dict[PilotStage, int] = {
    "setup": 7200,
    "baseline": 50400,
    "ablation": 14400,
    "evaluation": 14400,
}
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
SessionIdentifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class PilotSession(StrictRecord):
    """Count one instance session from requested startup until confirmed stop."""

    session_identifier: SessionIdentifier
    instance_identifier: str = Field(pattern=r"^i-[0-9a-f]{8,17}$")
    stage: PilotStage
    started_at: Seconds
    deadline_at: Seconds
    checkpoint_at: Seconds
    poweroff_at: Seconds
    last_observed_at: Seconds
    stopped_at: Seconds | None = None
    stop_confirmation: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def consistent_times(self) -> Self:
        """Reject impossible clocks and sessions with incomplete stop evidence."""
        if (
            not self.started_at
            < self.checkpoint_at
            < self.poweroff_at
            < self.deadline_at
        ):
            raise ValueError("Session deadlines must follow startup.")
        if self.deadline_at - self.started_at > STAGE_SECONDS[self.stage]:
            raise ValueError("The session exceeds its stage allocation.")
        if self.last_observed_at < self.started_at:
            raise ValueError("Observed time precedes startup.")
        if (self.stopped_at is None) != (self.stop_confirmation is None):
            raise ValueError("Stopped sessions require operator confirmation.")
        if self.stopped_at is not None and self.stopped_at < self.last_observed_at:
            raise ValueError("Confirmed stop cannot erase observed instance time.")
        return self


class PilotBudgetLedger(StrictRecord):
    """Retain all instance sessions without resetting the twenty-four-hour cap."""

    schema_version: Literal[1] = 1
    maximum_total_seconds: Literal[86400] = 86400
    checkpoint_reserve_seconds: Literal[300] = 300
    shutdown_reserve_seconds: Literal[120] = 120
    sessions: list[PilotSession] = Field(default_factory=list, max_length=256)

    @model_validator(mode="before")
    @classmethod
    def strict_fixed_numbers(cls, value: object) -> object:
        """Prevent bool/float equality from satisfying fixed accounting integers."""
        if isinstance(value, dict):
            for name in (
                "schema_version",
                "maximum_total_seconds",
                "checkpoint_reserve_seconds",
                "shutdown_reserve_seconds",
            ):
                if name in value and type(value[name]) is not int:
                    raise ValueError("Fixed accounting numbers require integers.")
        return value

    @model_validator(mode="after")
    def consistent_sessions(self) -> Self:
        """Require unique, non-overlapping sessions and one active instance."""
        names = [session.session_identifier for session in self.sessions]
        if len(names) != len(set(names)):
            raise ValueError("Session identifiers must be unique.")
        previous_stop = 0.0
        spent = 0.0
        stage_spent: dict[PilotStage, float] = {stage: 0.0 for stage in STAGE_SECONDS}
        for index, session in enumerate(self.sessions):
            if session.started_at < previous_stop:
                raise ValueError("Instance sessions cannot overlap.")
            if session.stopped_at is None and index != len(self.sessions) - 1:
                raise ValueError("Confirm the previous instance stopped first.")
            allowance = session.deadline_at - session.started_at
            if (
                allowance
                > min(
                    self.maximum_total_seconds - spent,
                    STAGE_SECONDS[session.stage] - stage_spent[session.stage],
                )
                + 1e-6
            ):
                raise ValueError("A session exceeds the remaining project allocation.")
            if (
                session.deadline_at - session.checkpoint_at
                != self.checkpoint_reserve_seconds
            ):
                raise ValueError("The fixed safe-save reserve cannot be changed.")
            if (
                session.deadline_at - session.poweroff_at
                != self.shutdown_reserve_seconds
            ):
                raise ValueError("The fixed shutdown reserve cannot be changed.")
            if session.stopped_at is not None:
                elapsed = session.stopped_at - session.started_at
                spent += elapsed
                stage_spent[session.stage] += elapsed
            previous_stop = session.stopped_at or session.started_at
        return self


class PilotBudgetSummary(StrictRecord):
    """Expose read-only spending information without starting an instance session."""

    initialized: bool
    total_seconds: Literal[86400] = 86400
    consumed_seconds: Seconds
    remaining_seconds: Seconds
    active_session_identifier: SessionIdentifier | None = None
