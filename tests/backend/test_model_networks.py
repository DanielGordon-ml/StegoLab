"""Small deterministic checks for the real neural model and integer channel."""

import torch
from torch import nn

from backend_service.model_networks import (
    build_models,
    model_pair_identifier,
    quantize_image,
)


def test_dense_shapes_seed_and_weight_identity() -> None:
    """Match the frozen topology and pair identity without consuming caller RNG."""
    initial_state = torch.get_rng_state().clone()
    encoder, decoder = build_models(8)
    assert torch.equal(torch.get_rng_state(), initial_state)
    matching_encoder, matching_decoder = build_models(8)
    assert model_pair_identifier(encoder, decoder) == model_pair_identifier(
        matching_encoder, matching_decoder
    )
    encoder_stages = [item for item in encoder.modules() if isinstance(item, nn.Conv2d)]
    decoder_stages = [item for item in decoder.modules() if isinstance(item, nn.Conv2d)]
    assert [(item.in_channels, item.out_channels) for item in encoder_stages] == [
        (3, 32),
        (33, 32),
        (65, 32),
        (97, 3),
    ]
    assert [(item.in_channels, item.out_channels) for item in decoder_stages] == [
        (3, 32),
        (32, 32),
        (64, 32),
        (96, 1),
    ]
    cover = torch.zeros((1, 3, 13, 17))
    payload = torch.ones((1, 1, 13, 17))
    raw = encoder(cover, payload)
    recovered = decoder(quantize_image(raw, straight_through=True))
    assert raw.shape == cover.shape
    assert recovered.shape == payload.shape
    recovered.square().mean().backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for model in (encoder, decoder)
        for parameter in model.parameters()
    )
    assert model_pair_identifier(encoder, decoder) != model_pair_identifier(
        matching_encoder, matching_decoder
    )


def test_rounding_forward_and_straight_through_clamp_gradient() -> None:
    """Use the same integer grid at training and inference, with no clipped gradient."""
    values = torch.tensor([-0.1, 0.25, 0.5, 0.75, 1.1], requires_grad=True)
    ordinary = quantize_image(values)
    approximate = quantize_image(values, straight_through=True)
    assert torch.equal(ordinary, approximate)
    assert torch.equal(
        torch.round(ordinary * 255), torch.tensor([0, 64, 128, 191, 255])
    )
    torch.autograd.backward(approximate.sum())
    assert values.grad is not None
    assert torch.equal(values.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))
