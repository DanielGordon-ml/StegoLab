"""Strict non-secret measurements for the experimental CPU model proof."""

from typing import Literal

from pydantic import ConfigDict, Field

from schemas.base import StrictRecord


class PngEvaluationTrial(StrictRecord):
    """Describe one attempted saved-image recovery without its secret contents."""

    model_config = ConfigDict(allow_inf_nan=False)
    source_identity: str = Field(min_length=1)
    width: int = Field(ge=512, le=1024)
    height: int = Field(ge=512, le=1024)
    fixture_kind: Literal["empty", "unicode", "maximum"]
    recovered: bool
    raw_bit_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    peak_signal_to_noise_ratio: float | None = None
    structural_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    identical_pixels: bool = False
    clipped_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    elapsed_seconds: float = Field(ge=0.0)
    error_code: str | None = None


class EvaluationReport(StrictRecord):
    """Separate completed attempts, exact recovery, and the unpromoted proof gate."""

    model_config = ConfigDict(allow_inf_nan=False)
    schema_version: Literal[1] = 1
    kind: Literal["proof", "validation"]
    compatibility_identifier: str
    dataset_revision: str
    completed: bool
    expected_bit_cases: int = Field(ge=0)
    bit_cases_completed: int = Field(ge=0)
    raw_bit_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_png_trials: int = Field(ge=1)
    png_trials: list[PngEvaluationTrial]
    exact_recovery_count: int = Field(ge=0)
    exact_recovery_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    median_peak_signal_to_noise_ratio: float | None = None
    median_structural_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    mean_clipped_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    elapsed_seconds: float = Field(ge=0.0)
    peak_process_memory_bytes: int = Field(ge=0)
    learning_gate_passed: bool
    stopped_reason: Literal["complete", "deadline", "failed"]
