"""Public integer-channel fixture pair for wrapper and browser tests.

This pair places message bits into the parity of the first colour channel and
reads them back. It is not a learned model and never measures image quality or
secrecy. Its only purpose is to exercise packaging, protocol, upload, and browser
paths with an exact channel that is cheap to build in tests and in CI.
"""

from pathlib import Path

import torch
from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.model_exports import export_models
from backend_service.model_networks import model_pair_identifier
from schemas.model_exports import ExportMetadata, ModelExportSummary

FIXTURE_SOURCE_IDENTIFIER = "public_integer_channel_fixture_not_a_neural_model"
FIXTURE_PACKAGE_NAME = "fixture_not_a_neural_model"


class IntegerChannelEncoder(nn.Module):
    """Use a public integer channel to test wrappers, never learned quality."""

    def forward(self, image: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        """Place each bit in the first channel's integer parity."""
        first = (torch.floor(image[:, :1] * 255 / 2) * 2 + payload) / 255
        return torch.cat((first, image[:, 1:]), dim=1)


class IntegerChannelDecoder(nn.Module):
    """Recover the public fixture's integer parity as signed logits."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract the fixture channel without any encoder dependency."""
        return (torch.remainder(torch.round(image[:, :1] * 255), 2) - 0.5) * 20


def build_fixture_package(
    destination: Path, *, deadline: float | None = None
) -> ModelExportSummary:
    """Export the labelled fixture pair as an ordinary verified package directory."""
    try:
        # A fresh checkout has no models folder yet; the export needs its parent.
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ApplicationFailure(
            "fixture_destination_unavailable",
            "The folder for the fixture package could not be created. Check the "
            "destination path and its access rights.",
            422,
        ) from None
    encoder, decoder = IntegerChannelEncoder(), IntegerChannelDecoder()
    metadata = ExportMetadata(
        compatibility_identifier=model_pair_identifier(encoder, decoder),
        source_identifier=FIXTURE_SOURCE_IDENTIFIER,
    )
    return export_models(encoder, decoder, destination, metadata, deadline=deadline)
