"""Synchronous shuffled crops with exact consumed-position restoration."""

import hashlib
import random
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from backend_service.dataset_reader import DatasetExample
from backend_service.dataset_serialization import read_bounded
from backend_service.failures import ApplicationFailure
from backend_service.pilot_data import (
    PilotData,
    check_pilot_deadline,
    pilot_data_failure,
    source_identity,
)
from schemas.pilot_data import PilotRandomState, PilotSample, PilotSamplerState


def _load_crop(example: DatasetExample, left: int, top: int) -> Tensor:
    """Verify bounded PNG bytes before decoding an image and extracting its crop."""
    record = example.record
    try:
        content = read_bounded(example.path, record.prepared_bytes)
        if (
            len(content) != record.prepared_bytes
            or hashlib.sha256(content).hexdigest() != record.prepared_checksum
        ):
            raise ValueError
        with Image.open(BytesIO(content), formats=["PNG"]) as image:
            if image.mode != record.mode or image.size != (record.width, record.height):
                raise ValueError
            with image.crop((left, top, left + 256, top + 256)) as crop:
                with crop.convert("RGB") as rgb:
                    pixels = np.array(rgb, dtype=np.uint8, copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float() / 255.0
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError):
        raise pilot_data_failure() from None


class PilotSampler:
    """Consume one shuffled epoch at a time, with no workers or hidden prefetch."""

    def __init__(
        self, data: PilotData, seed: int, *, deadline: float | None = None
    ) -> None:
        """Start independent image/crop randomness without affecting payload bits."""
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ValueError("The sampler seed must be a nonnegative bounded integer.")
        self.data = data
        self.seed = seed
        self.deadline = deadline
        self.random = random.Random(seed)
        self.order = list(range(len(data.training_examples)))
        if not self.order:
            raise pilot_data_failure()
        self.random.shuffle(self.order)
        self.epoch = 0
        self.position = 0
        self.consumed = 0
        self.last_batch: tuple[PilotSample, ...] = ()

    def state_dict(self) -> dict[str, object]:
        """Return a validated snapshot containing only safe primitive values."""
        state = PilotSamplerState(
            dataset_revision=self.data.manifest.revision,
            selection_checksum=self.data.selection.selection_checksum,
            seed=self.seed,
            epoch=self.epoch,
            position=self.position,
            consumed=self.consumed,
            order=self.order.copy(),
            random_state=PilotRandomState(values=list(self.random.getstate()[1])),
        )
        return state.model_dump(mode="json")

    def load_state_dict(self, value: object) -> None:
        """Reject incompatible snapshots before changing the live sampler state."""
        try:
            state = PilotSamplerState.model_validate(value)
            count = len(self.data.training_examples)
            if (
                state.dataset_revision != self.data.manifest.revision
                or state.selection_checksum != self.data.selection.selection_checksum
                or state.seed != self.seed
                or len(state.order) != count
                or sorted(state.order) != list(range(count))
                or state.position > count
                or state.consumed != state.epoch * count + state.position
            ):
                raise ValueError
            generator = random.Random()
            generator.setstate((3, tuple(state.random_state.values), None))
        except (ValueError, TypeError, OverflowError):
            raise ApplicationFailure(
                "pilot_sampler_invalid",
                "The saved sample position is incompatible. Use its original dataset "
                "and seed, or start a new pilot run.",
                422,
            ) from None
        self.random = generator
        self.order = state.order.copy()
        self.epoch = state.epoch
        self.position = state.position
        self.consumed = state.consumed
        self.last_batch = ()

    def next_batch(self, batch_size: int) -> Tensor:
        """Read exactly the requested crops and publish progress only on success."""
        if type(batch_size) is not int or not 1 <= batch_size <= 16:
            raise ValueError("The synchronous batch size must be between 1 and 16.")
        previous = self.state_dict()
        previous_batch = self.last_batch
        crops: list[Tensor] = []
        samples: list[PilotSample] = []
        try:
            for _ in range(batch_size):
                check_pilot_deadline(self.deadline)
                if self.position == len(self.order):
                    self.random.shuffle(self.order)
                    self.epoch += 1
                    self.position = 0
                example = self.data.training_examples[self.order[self.position]]
                left = self.random.randrange(example.record.width - 256 + 1)
                top = self.random.randrange(example.record.height - 256 + 1)
                crops.append(_load_crop(example, left, top))
                samples.append(
                    PilotSample(
                        source_identity=source_identity(example.record),
                        left=left,
                        top=top,
                    )
                )
                self.position += 1
                self.consumed += 1
            check_pilot_deadline(self.deadline)
            result = torch.stack(crops)
        except BaseException:
            self.load_state_dict(previous)
            self.last_batch = previous_batch
            raise
        self.last_batch = tuple(samples)
        return result
