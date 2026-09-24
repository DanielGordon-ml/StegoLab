"""Real later GPU probes for deterministic interrupted training continuation."""

import copy
import time
from pathlib import Path
from typing import Any, Literal

import torch

from backend_service.failures import ApplicationFailure
from backend_service.pilot_runtime import (
    check_pilot_deadline,
    configure_pilot,
    synchronize_device,
)
from backend_service.pilot_training_state import (
    PilotTrainingSession,
    create_pilot_session,
    pilot_training_step,
    save_pilot_session,
)
from backend_service.proof_runtime import write_record
from backend_service.training_checkpoint_state import random_state
from schemas.pilot_readiness import GpuReadinessRequest, GpuResumeEvidence
from schemas.pilot_training import PilotTrainingConfiguration, PilotTrainingRequest


def probe_cuda_device() -> dict[str, str]:
    """Initialize exactly one supported CUDA device and execute a real operation."""
    if str(torch.__version__) != "2.14.0+cu126" or torch.version.cuda != "12.6":
        raise ApplicationFailure(
            "pilot_cuda_environment",
            "Use the locked PyTorch 2.14.0 CUDA 12.6 readiness environment.",
            422,
        )
    selected = configure_pilot(
        PilotTrainingConfiguration(device="cuda", planned_optimizer_steps=2)
    )
    values = torch.ones((16, 16), device=selected)
    result = values @ values
    synchronize_device(selected)
    if not torch.equal(result.cpu(), torch.full((16, 16), 16.0)):
        raise ApplicationFailure(
            "pilot_cuda_probe", "The CUDA arithmetic probe failed."
        )
    return {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(selected),
        "total_memory_bytes": str(
            torch.cuda.get_device_properties(selected).total_memory
        ),
    }


def _equal(expected: Any, actual: Any) -> bool:
    """Compare safe tensor trees exactly, including optimizer and random state."""
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and torch.equal(expected, actual)
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _equal(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return len(expected) == len(actual) and all(
            _equal(left, right) for left, right in zip(expected, actual, strict=True)
        )
    return bool(expected == actual)


def _snapshot(session: PilotTrainingSession) -> dict[str, Any]:
    """Freeze the exact state immediately after a completed optimizer update."""
    return copy.deepcopy(
        {
            "encoder": session.encoder.state_dict(),
            "decoder": session.decoder.state_dict(),
            "optimizer": session.optimizer.state_dict(),
            "random": random_state(),
            "sampler": session.sampler.state_dict(),
            "step": session.global_step,
        }
    )


def verify_cuda_resume(
    request: GpuReadinessRequest,
    directory: Path,
    batch_size: Literal[4, 8],
    deadline: float,
) -> list[str]:
    """Compare two continuous updates with one saved update and one resumed update."""
    configuration = PilotTrainingConfiguration(
        device="cuda", physical_batch_size=batch_size, planned_optimizer_steps=2
    )
    training = PilotTrainingRequest(
        experiment_identifier=f"readiness_batch_{batch_size}",
        dataset_directory=request.dataset_directory,
        output_root=str(directory),
        execution_mode="gpu_pilot",
        session_identifier=request.session_identifier,
        configuration=configuration,
    )
    continuous = create_pilot_session(training, deadline=deadline)
    torch.cuda.reset_peak_memory_stats(continuous.device)
    started = time.monotonic()
    for _ in range(2):
        check_pilot_deadline(deadline)
        pilot_training_step(continuous)
    synchronize_device(continuous.device)
    continuous_seconds = time.monotonic() - started
    expected = _snapshot(continuous)
    expected_payload = torch.randint(0, 2, (8,), device=continuous.device)
    expected_crop = continuous.sampler.next_batch(1)
    expected_sample = continuous.sampler.last_batch
    del continuous
    interrupted = create_pilot_session(training, deadline=deadline)
    check_pilot_deadline(deadline)
    pilot_training_step(interrupted)
    summary = save_pilot_session(interrupted, directory / "checkpoints")
    checkpoint = directory / "checkpoints" / summary.checkpoint_identifier
    del interrupted
    resumed = create_pilot_session(
        training.model_copy(update={"resume_checkpoint": str(checkpoint)}),
        deadline=deadline,
    )
    check_pilot_deadline(deadline)
    started = time.monotonic()
    pilot_training_step(resumed)
    synchronize_device(resumed.device)
    resumed_seconds = time.monotonic() - started
    if (
        resumed.sampler.consumed != 32
        or not _equal(expected, _snapshot(resumed))
        or not torch.equal(
            expected_payload, torch.randint(0, 2, (8,), device=resumed.device)
        )
        or not torch.equal(expected_crop, resumed.sampler.next_batch(1))
        or expected_sample != resumed.sampler.last_batch
    ):
        raise ApplicationFailure(
            "pilot_resume_mismatch",
            "Continuous and resumed GPU state differ. Keep this profile blocked.",
            422,
        )
    check_pilot_deadline(deadline)
    destination = directory / "resume_verification.json"
    write_record(
        destination,
        GpuResumeEvidence(
            physical_batch_size=batch_size,
            uninterrupted_training_seconds=continuous_seconds,
            resumed_update_seconds=resumed_seconds,
            peak_device_memory_bytes=int(
                torch.cuda.max_memory_allocated(resumed.device)
            ),
            checkpoint=str(checkpoint),
        ),
    )
    return [str(checkpoint), str(destination)]
