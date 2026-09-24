"""Real pilot batches, frozen sampling state, and exact checkpoint continuation."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.model_networks import build_models, quantize_image
from backend_service.pilot_data import PilotData, load_pilot_data
from backend_service.pilot_runtime import (
    configure_pilot,
    pilot_environment,
    pilot_source_identity,
)
from backend_service.pilot_sampler import PilotSampler
from backend_service.training_checkpoints import load_checkpoint, save_checkpoint
from schemas.checkpoints import CheckpointMetrics, CheckpointSummary
from schemas.pilot_training import PilotTrainingConfiguration, PilotTrainingRequest


def selection_identifier(data: PilotData) -> str:
    """Bind the exact frozen cover selection to each checkpoint and report."""
    return data.selection.selection_checksum


@dataclass
class PilotTrainingSession:
    """Keep mutable training state separate from requests and durable reports."""

    encoder: nn.Module
    decoder: nn.Module
    optimizer: torch.optim.Adam
    data: PilotData
    sampler: PilotSampler
    configuration: PilotTrainingConfiguration
    identities: dict[str, str]
    device: torch.device
    global_step: int = 0


def create_pilot_session(
    request: PilotTrainingRequest,
    *,
    deadline: float | None = None,
) -> PilotTrainingSession:
    """Load metadata, select the explicit device and restore only compatible state."""
    configuration = request.configuration
    device = configure_pilot(configuration)
    data = load_pilot_data(Path(request.dataset_directory), deadline=deadline)
    if (
        request.execution_mode == "gpu_pilot"
        and not data.selection.gpu_profile_eligible
    ):
        raise ApplicationFailure(
            "pilot_dataset_ineligible",
            "The GPU profile requires the frozen full UHD-IQA revision and selection.",
            422,
        )
    encoder, decoder = build_models(configuration.seed)
    encoder.to(device)
    decoder.to(device)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()],
        lr=configuration.learning_rate,
    )
    identities = {
        "dataset_revision": data.manifest.revision,
        "selection": selection_identifier(data),
        "environment": pilot_environment(configuration),
        "architecture": configuration.architecture,
        "experiment_identifier": request.experiment_identifier,
        "training_source": pilot_source_identity(),
    }
    session = PilotTrainingSession(
        encoder,
        decoder,
        optimizer,
        data,
        PilotSampler(data, configuration.seed),
        configuration,
        identities,
        device,
    )
    if request.resume_checkpoint is not None:
        loaded = load_checkpoint(
            Path(request.resume_checkpoint),
            encoder,
            decoder,
            optimizer,
            identities=identities,
            configuration=configuration.model_dump(mode="json"),
        )
        session.sampler.load_state_dict(loaded.sampler_state)
        session.global_step = loaded.summary.global_step
        if (
            session.global_step > configuration.planned_optimizer_steps
            or session.sampler.consumed
            != session.global_step * configuration.effective_batch_size
        ):
            raise ApplicationFailure(
                "checkpoint_incompatible",
                "The saved sample count or step conflicts with the frozen schedule.",
            )
    return session


def pilot_training_step(session: PilotTrainingSession) -> tuple[float, float, float]:
    """Commit sixteen samples in fixed physical batches without changing BatchNorm."""
    configuration = session.configuration
    weight = configuration.maximum_image_loss_weight * min(
        session.global_step / configuration.image_loss_ramp_steps,
        1.0,
    )
    session.encoder.train()
    session.decoder.train()
    session.optimizer.zero_grad(set_to_none=True)
    accumulation = (
        configuration.effective_batch_size // configuration.physical_batch_size
    )
    bit_total = image_total = 0.0
    for _ in range(accumulation):
        cover = session.sampler.next_batch(configuration.physical_batch_size).to(
            session.device
        )
        payload = torch.randint(
            0,
            2,
            (configuration.physical_batch_size, 1, 256, 256),
            device=session.device,
        ).float()
        raw = session.encoder(cover, payload)
        quantized = quantize_image(raw, straight_through=True)
        logits = session.decoder(quantized)
        bit_loss = nn.functional.binary_cross_entropy_with_logits(logits, payload)
        image_loss = nn.functional.mse_loss(quantized, cover)
        loss = (bit_loss + weight * image_loss) / accumulation
        if not bool(torch.isfinite(loss)):
            raise ApplicationFailure(
                "training_nonfinite",
                "Training produced invalid values; inspect the last completed "
                "checkpoint.",
            )
        torch.autograd.backward(loss)
        bit_total += float(bit_loss.detach()) / accumulation
        image_total += float(image_loss.detach()) / accumulation
    session.optimizer.step()
    session.global_step += 1
    return bit_total, image_total, weight


def save_pilot_session(
    session: PilotTrainingSession,
    directory: Path,
    metrics: CheckpointMetrics | None = None,
) -> CheckpointSummary:
    """Save consumed samples and random state only at a complete update boundary."""
    return save_checkpoint(
        directory,
        session.encoder,
        session.decoder,
        session.optimizer,
        global_step=session.global_step,
        sampler_state=session.sampler.state_dict(),
        configuration=session.configuration.model_dump(mode="json"),
        identities=session.identities,
        metrics=metrics,
    )
