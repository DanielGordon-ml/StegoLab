"""Small deterministic CPU trainer with bounded, explicit checkpoint resume."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn

from backend_service.failures import ApplicationFailure
from backend_service.model_data import ProofData, load_proof_data
from backend_service.model_evaluation import evaluate_models
from backend_service.model_networks import (
    build_models,
    model_pair_identifier,
    quantize_image,
)
from backend_service.proof_runtime import (
    StopRequest,
    append_record,
    check_output_root,
    configure_cpu,
    environment_identity,
    run_directory,
    training_source_identity,
    write_record,
)
from backend_service.training_budget import ProofBudget
from backend_service.training_checkpoints import load_checkpoint, save_checkpoint
from schemas.checkpoints import CheckpointMetrics, CheckpointSummary
from schemas.training import (
    TrainingConfiguration,
    TrainingRequest,
    TrainingRun,
    TrainingStep,
)


@dataclass
class TrainingSession:
    """Keep in-memory learning state separate from serialized request paths."""

    encoder: nn.Module
    decoder: nn.Module
    optimizer: torch.optim.Adam
    data: ProofData
    configuration: TrainingConfiguration
    identities: dict[str, str]
    global_step: int = 0


def create_session(
    request: TrainingRequest, *, deadline: float | None = None
) -> TrainingSession:
    """Validate one revision once and restore only explicitly selected state."""
    configure_cpu(request.configuration)
    data = load_proof_data(Path(request.dataset_directory), deadline=deadline)
    encoder, decoder = build_models(request.configuration.seed)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()],
        lr=request.configuration.learning_rate,
    )
    identities = {
        "dataset_revision": data.manifest.revision,
        "environment": environment_identity(request.configuration),
        "architecture": request.configuration.architecture,
        "experiment_identifier": request.experiment_identifier,
        "training_source": training_source_identity(),
    }
    session = TrainingSession(
        encoder, decoder, optimizer, data, request.configuration, identities
    )
    if request.resume_checkpoint is not None:
        loaded = load_checkpoint(
            Path(request.resume_checkpoint),
            encoder,
            decoder,
            optimizer,
            identities=identities,
            configuration=request.configuration.model_dump(mode="json"),
        )
        session.global_step = loaded.summary.global_step
        if loaded.sampler_state != {"next_cover": 0}:
            raise ApplicationFailure(
                "checkpoint_incompatible", "The checkpoint sampling state is invalid."
            )
    return session


def training_step(session: TrainingSession) -> tuple[float, float, float]:
    """Accumulate four physical batches and commit one complete optimizer step."""
    configuration = session.configuration
    weight = configuration.maximum_image_loss_weight * min(
        session.global_step / configuration.image_loss_ramp_steps, 1.0
    )
    session.encoder.train()
    session.decoder.train()
    session.optimizer.zero_grad(set_to_none=True)
    bit_total = image_total = 0.0
    for source in session.data.training_crops:
        cover = source.unsqueeze(0)
        payload = torch.randint(0, 2, (1, 1, 256, 256)).to(torch.float32)
        predicted: Tensor = session.encoder(cover, payload)
        quantized = quantize_image(predicted, straight_through=True)
        logits: Tensor = session.decoder(quantized)
        bit_loss = nn.functional.binary_cross_entropy_with_logits(logits, payload)
        image_loss = nn.functional.mse_loss(quantized, cover)
        loss = (bit_loss + weight * image_loss) / configuration.effective_batch_size
        if not torch.isfinite(loss).item():
            raise ApplicationFailure(
                "training_nonfinite", "Training produced invalid values; check the run."
            )
        torch.autograd.backward(loss)
        bit_total += float(bit_loss.detach()) / configuration.effective_batch_size
        image_total += float(image_loss.detach()) / configuration.effective_batch_size
    session.optimizer.step()
    session.global_step += 1
    return bit_total, image_total, weight


def _save(
    session: TrainingSession,
    directory: Path,
    metrics: CheckpointMetrics | None = None,
) -> CheckpointSummary:
    """Save only completed accumulation steps with no partial gradients."""
    return save_checkpoint(
        directory,
        session.encoder,
        session.decoder,
        session.optimizer,
        global_step=session.global_step,
        sampler_state={"next_cover": 0},
        configuration=session.configuration.model_dump(mode="json"),
        identities=session.identities,
        metrics=metrics,
    )


def _validate(
    session: TrainingSession, deadline: float, logs: Path
) -> CheckpointMetrics | None:
    """Score a fixed tuning suite without consuming the training random stream."""
    report = evaluate_models(
        session.encoder,
        session.decoder,
        session.data,
        model_pair_identifier(session.encoder, session.decoder),
        deadline=deadline,
        lightweight=True,
    )
    write_record(logs / f"validation_{session.global_step:04d}.json", report)
    if not report.completed:
        return None
    return CheckpointMetrics(
        exact_message_recovery=report.exact_recovery_rate or 0.0,
        median_psnr=report.median_peak_signal_to_noise_ratio or 0.0,
        median_ssim=report.median_structural_similarity or 0.0,
    )


def train(request: TrainingRequest) -> TrainingRun:
    """Run one bounded CPU experiment and always save at a normal stop boundary."""
    root = Path(request.output_root).resolve()
    check_output_root(root, Path(request.dataset_directory).resolve())
    checkpoint_directory = root / "checkpoints" / request.experiment_identifier
    with (
        ProofBudget(
            root / "state" / "cpu_proof",
            request.experiment_identifier,
            resume=request.resume_checkpoint is not None,
            checkpoint_directory=checkpoint_directory,
        ) as budget,
        StopRequest() as stop,
    ):
        started = time.monotonic()
        logs = run_directory(root, request.experiment_identifier)
        session = create_session(
            request, deadline=time.monotonic() + budget.training_remaining_seconds
        )
        if session.global_step > request.stop_after_step:
            raise ApplicationFailure(
                "training_steps", "The stop step is earlier than the saved checkpoint."
            )
        status: Literal["completed", "stopped", "budget_exhausted"] = "completed"
        saved_at = time.monotonic()
        while session.global_step < request.stop_after_step:
            if stop.signum is not None:
                status = "stopped"
                break
            if budget.training_remaining_seconds <= 0:
                status = "budget_exhausted"
                break
            bit_loss, image_loss, weight = training_step(session)
            progress = TrainingStep(
                global_step=session.global_step,
                bit_loss=bit_loss,
                image_loss=image_loss,
                image_loss_weight=weight,
                elapsed_seconds=time.monotonic() - started,
            )
            append_record(logs / "steps.jsonl", progress)
            due = (
                session.global_step % session.configuration.validation_interval_steps
                == 0
            )
            metrics = None
            if due and stop.signum is None and budget.training_remaining_seconds > 0:
                deadline = time.monotonic() + budget.training_remaining_seconds
                metrics = _validate(session, deadline, logs)
            if due or time.monotonic() - saved_at >= 300:
                _save(session, checkpoint_directory, metrics)
                saved_at = time.monotonic()
        if stop.signum is not None:
            status = "stopped"
        checkpoint = _save(session, checkpoint_directory)
        result = TrainingRun(
            experiment_identifier=request.experiment_identifier,
            status=status,
            global_step=session.global_step,
            checkpoint=str(
                (checkpoint_directory / checkpoint.checkpoint_identifier).relative_to(
                    root
                )
            ),
            dataset_revision=session.data.manifest.revision,
            compatibility_identifier=model_pair_identifier(
                session.encoder, session.decoder
            ),
            elapsed_seconds=time.monotonic() - started,
            remaining_experiment_seconds=max(0.0, budget.remaining_seconds),
            selected_training_identities=list(session.data.identities["train"]),
            selected_tuning_identities=list(session.data.identities["tuning"]),
            stop_signal=cast(Literal[2, 15] | None, stop.signum),
        )
        write_record(logs / "training_run.json", result)
        return result
