"""Bounded smoke or GPU training without public model promotion."""

import time
from pathlib import Path
from typing import Literal, cast

from backend_service.failures import ApplicationFailure
from backend_service.model_networks import model_pair_identifier
from backend_service.pilot_evaluation import evaluate_pilot_models
from backend_service.pilot_execution import pilot_execution
from backend_service.pilot_runtime import pilot_experiment_lock, synchronize_device
from backend_service.pilot_training_state import (
    create_pilot_session,
    pilot_training_step,
    save_pilot_session,
)
from backend_service.proof_runtime import (
    StopRequest,
    append_record,
    check_output_root,
    run_directory,
    write_record,
)
from schemas.checkpoints import CheckpointMetrics
from schemas.pilot_training import (
    PilotTrainingRequest,
    PilotTrainingRun,
    PilotTrainingStep,
)


def train_pilot(request: PilotTrainingRequest) -> PilotTrainingRun:
    """Run a fixed profile under the independent smoke or GPU deadline."""
    request = PilotTrainingRequest.model_validate(request)
    root = Path(request.output_root).resolve()
    check_output_root(root, Path(request.dataset_directory).resolve())
    directory = root / "checkpoints" / request.experiment_identifier
    with pilot_experiment_lock(root, request.experiment_identifier):
        return _train_locked(request, root, directory)


def _train_locked(
    request: PilotTrainingRequest, root: Path, directory: Path
) -> PilotTrainingRun:
    """Own the experiment through startup, complete steps and the final safe save."""
    if (
        request.resume_checkpoint is None
        and directory.exists()
        and any(directory.iterdir())
    ):
        raise ApplicationFailure(
            "pilot_resume_required",
            "This experiment already has saved state. "
            "Provide its checkpoint to resume or choose a new experiment identifier.",
            422,
        )
    with pilot_execution(request) as deadline, StopRequest() as stop:
        started = time.monotonic()
        logs = run_directory(root, request.experiment_identifier)
        session = create_pilot_session(request, deadline=deadline)
        target = (
            request.stop_after_step or session.configuration.planned_optimizer_steps
        )
        if session.global_step > target:
            raise ApplicationFailure(
                "training_steps",
                "The stop step is earlier than the saved checkpoint.",
                422,
            )
        status: Literal["completed", "stopped", "budget_exhausted"] = "completed"
        saved_at = time.monotonic()
        selected = False
        while session.global_step < target:
            if stop.signum is not None:
                status = "stopped"
                break
            if time.monotonic() >= deadline:
                status = "budget_exhausted"
                break
            bit_loss, image_loss, weight = pilot_training_step(session)
            synchronize_device(session.device)
            append_record(
                logs / "steps.jsonl",
                PilotTrainingStep(
                    global_step=session.global_step,
                    bit_loss=bit_loss,
                    image_loss=image_loss,
                    image_loss_weight=weight,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            metrics = None
            due = (
                session.global_step % session.configuration.validation_interval_steps
                == 0
            )
            if (
                request.execution_mode == "gpu_pilot"
                and due
                and stop.signum is None
                and time.monotonic() < deadline
            ):
                report = evaluate_pilot_models(
                    session.encoder,
                    session.decoder,
                    session.data,
                    model_pair_identifier(session.encoder, session.decoder),
                    device=session.device,
                    deadline=deadline,
                )
                write_record(
                    logs / f"validation_{session.global_step:08d}.json", report
                )
                if report.completed:
                    metrics = CheckpointMetrics(
                        exact_message_recovery=report.exact_recovery_rate or 0.0,
                        median_psnr=report.median_peak_signal_to_noise_ratio or 0.0,
                        median_ssim=report.median_structural_similarity or 0.0,
                    )
                    selected = True
            if metrics is not None or time.monotonic() - saved_at >= 300:
                save_pilot_session(session, directory, metrics)
                saved_at = time.monotonic()
        if stop.signum is not None:
            status = "stopped"
        checkpoint = save_pilot_session(session, directory)
        result = PilotTrainingRun(
            experiment_identifier=request.experiment_identifier,
            execution_mode=request.execution_mode,
            status=status,
            global_step=session.global_step,
            checkpoint=str(
                (directory / checkpoint.checkpoint_identifier).relative_to(root)
            ),
            dataset_revision=session.data.manifest.revision,
            compatibility_identifier=model_pair_identifier(
                session.encoder, session.decoder
            ),
            elapsed_seconds=time.monotonic() - started,
            quality_selection_performed=selected,
            stop_signal=cast(Literal[2, 15] | None, stop.signum),
        )
        write_record(logs / "training_run.json", result)
        return result
