"""Verify real dense continuation and strictly bounded CPU preparation behavior."""

import json
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from pilot_data_fixtures import pilot_revision
from pydantic import ValidationError

from backend_service.failures import ApplicationFailure
from backend_service.pilot_runtime import configure_pilot, pilot_experiment_lock
from backend_service.pilot_training import train_pilot
from backend_service.pilot_training_state import (
    PilotTrainingSession,
    StepLosses,
    create_pilot_session,
    pilot_training_step,
    save_pilot_session,
)
from backend_service.training_checkpoints import read_checkpoint_state
from schemas.pilot_training import (
    PilotEvaluationRequest,
    PilotExportRequest,
    PilotTrainingConfiguration,
    PilotTrainingRequest,
)


def request(root: Path, batch_size: Literal[4, 8] = 4) -> PilotTrainingRequest:
    """Use an immutable synthetic revision and one CPU thread."""
    directory = pilot_revision(root / "dataset")
    return PilotTrainingRequest(
        experiment_identifier="pilot_test",
        output_root=str(root / "output"),
        dataset_directory=str(directory),
        configuration=PilotTrainingConfiguration(
            planned_optimizer_steps=10,
            physical_batch_size=batch_size,
            cpu_threads=1,
        ),
        stop_after_step=2,
    )


@pytest.mark.parametrize("batch_size", [4, 8])
def test_real_dense_resume_matches_within_each_physical_batch(
    tmp_path: Path,
    batch_size: Literal[4, 8],
) -> None:
    """Restore Adam, BatchNorm, image/crop order and payload randomness exactly."""
    original = request(tmp_path, batch_size)
    continuous = create_pilot_session(original)
    pilot_training_step(continuous)
    expected_loss = pilot_training_step(continuous)
    expected_bits = torch.rand(8)
    expected_crops = continuous.sampler.next_batch(1)
    expected_samples = continuous.sampler.last_batch
    interrupted = create_pilot_session(original)
    pilot_training_step(interrupted)
    saved = save_pilot_session(interrupted, tmp_path / "checkpoints")
    resumed = create_pilot_session(
        original.model_copy(
            update={
                "resume_checkpoint": str(
                    tmp_path / "checkpoints" / saved.checkpoint_identifier
                ),
            }
        )
    )
    assert resumed.global_step == 1
    assert pilot_training_step(resumed) == expected_loss
    assert resumed.sampler.consumed == 32
    for expected, actual in (
        (continuous.encoder, resumed.encoder),
        (continuous.decoder, resumed.decoder),
    ):
        for name, values in expected.state_dict().items():
            assert torch.equal(values, actual.state_dict()[name]), name
    expected_state = continuous.optimizer.state_dict()["state"]
    for key, values in resumed.optimizer.state_dict()["state"].items():
        for name, value in values.items():
            assert torch.equal(value, expected_state[key][name])
    assert torch.equal(torch.rand(8), expected_bits)
    assert torch.equal(resumed.sampler.next_batch(1), expected_crops)
    assert resumed.sampler.last_batch == expected_samples
    assert resumed.sampler.state_dict() == continuous.sampler.state_dict()


def test_smoke_signal_saves_complete_step_without_quality_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM waits for all sixteen samples, then saves unranked smoke state."""
    original = request(tmp_path)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted_step(session: PilotTrainingSession) -> StepLosses:
        """Signal during one update while allowing its accumulation to finish."""
        signal.raise_signal(signal.SIGTERM)
        return pilot_training_step(session)

    def forbidden_quality(*arguments: object, **keywords: object) -> None:
        """Prevent an unrequested scientific evaluation during CPU smoke work."""
        raise AssertionError("CPU smoke must not select quality candidates.")

    monkeypatch.setattr(
        "backend_service.pilot_training.pilot_training_step", interrupted_step
    )
    monkeypatch.setattr(
        "backend_service.pilot_training.evaluate_pilot_models", forbidden_quality
    )
    result = train_pilot(original)
    assert result.global_step == 1 and result.status == "stopped"
    assert result.stop_signal == 15 and not result.quality_selection_performed
    assert signal.getsignal(signal.SIGTERM) == previous
    saved = Path(original.output_root) / result.checkpoint
    metadata = json.loads((saved / "metadata.json").read_text())
    assert metadata["metrics"] is None
    assert not (Path(original.output_root) / "state" / "gpu_pilot").exists()
    assert not (Path(original.output_root) / "state" / "cpu_proof").exists()
    with pytest.raises(ApplicationFailure, match="already has saved state"):
        train_pilot(original)


def test_pilot_schema_bounds_and_schedule_are_frozen(tmp_path: Path) -> None:
    """Reject request ambiguity and derive the ramp from planned, not stop, steps."""
    original = request(tmp_path)
    configuration = original.configuration
    assert configuration.image_loss_ramp_steps == 2
    assert (
        PilotTrainingConfiguration(planned_optimizer_steps=11).image_loss_ramp_steps
        == 3
    )
    for change in (
        {"stop_after_step": 11},
        {"schema_version": 2.0},
        {"execution_mode": "gpu_pilot"},
        {"session_identifier": "unexpected"},
    ):
        with pytest.raises(ValidationError):
            PilotTrainingRequest.model_validate(original.model_dump() | change)
    for configuration_change in (
        {"physical_batch_size": True},
        {"physical_batch_size": 4.0},
        {"seed": True},
        {"effective_batch_size": 8},
    ):
        with pytest.raises(ValidationError):
            PilotTrainingConfiguration.model_validate(
                configuration.model_dump() | configuration_change
            )
    with pytest.raises(ValidationError):
        PilotEvaluationRequest(
            experiment_identifier="test",
            checkpoint="saved",
            dataset_directory="data",
            device="cuda",
        )
    with pytest.raises(ValidationError):
        PilotExportRequest(
            experiment_identifier="test", checkpoint="saved", device="cuda"
        )


def test_smoke_experiment_lock_rejects_concurrent_writers(tmp_path: Path) -> None:
    """A second process lock cannot pass the initial state check during smoke."""
    with pilot_experiment_lock(tmp_path, "shared_run"):
        with pytest.raises(ApplicationFailure) as caught:
            with pilot_experiment_lock(tmp_path, "shared_run"):
                raise AssertionError("A concurrent writer entered the experiment.")
        assert caught.value.code == "pilot_experiment_busy"
    with pilot_experiment_lock(tmp_path, "shared_run"):
        assert (tmp_path / "state" / "pilot_operations" / "shared_run.lock").is_file()


def test_expired_work_deadline_saves_the_last_complete_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiration after an update preserves sixteen consumed samples and weights."""
    original = request(tmp_path)
    session = create_pilot_session(original)
    clock = [100.0]

    @contextmanager
    def allowance(value: PilotTrainingRequest) -> Iterator[float]:
        """Provide one controlled work deadline without sleeping or GPU accounting."""
        yield 101.0

    def completing_step(active: PilotTrainingSession) -> StepLosses:
        """Expire the work allowance immediately after a real committed update."""
        losses = pilot_training_step(active)
        clock[0] = 102.0
        return losses

    monkeypatch.setattr("backend_service.pilot_training.pilot_execution", allowance)
    monkeypatch.setattr(
        "backend_service.pilot_training.create_pilot_session",
        lambda *arguments, **keywords: session,
    )
    monkeypatch.setattr(
        "backend_service.pilot_training.time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    monkeypatch.setattr(
        "backend_service.pilot_training.pilot_training_step", completing_step
    )
    result = train_pilot(original)
    assert result.status == "budget_exhausted" and result.global_step == 1
    saved = read_checkpoint_state(Path(original.output_root) / result.checkpoint)
    assert saved["global_step"] == 1 and saved["sampler_state"]["consumed"] == 16
    for name, value in session.encoder.state_dict().items():
        assert torch.equal(saved["encoder"][name], value)


def test_cuda_request_fails_without_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU-only host must not run an accidentally downgraded GPU experiment."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ApplicationFailure) as caught:
        configure_pilot(
            PilotTrainingConfiguration(planned_optimizer_steps=10, device="cuda")
        )
    assert caught.value.code == "pilot_cuda_unavailable"
