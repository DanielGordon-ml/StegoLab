"""Verify a CUDA image on a native CPU runner without claiming GPU acceptance."""

import copy
import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path

import torch

from backend_service.failures import ApplicationFailure
from backend_service.model_networks import build_models, quantize_image
from backend_service.pilot_runtime import require_device
from backend_service.training_checkpoint_state import random_state, restore_random_state


def main() -> int:
    """Check CUDA packaging and small CPU forward/resume/export execution."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Use the native Linux amd64 preparation runner.")
    if torch.__version__ != "2.14.0+cu126" or torch.version.cuda != "12.6":
        raise RuntimeError("The locked CUDA runtime is not installed.")
    path = Path("infrastructure/cuda/runtime_dependencies.json")
    versions = json.loads(path.read_text(encoding="utf-8"))
    for package, version in versions.items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError("The installed runtime differs from its locked record.")
    if torch.cuda.is_available():
        raise RuntimeError("This preparation check must run without GPU access.")
    try:
        require_device("cuda")
    except ApplicationFailure as failure:
        if failure.code != "pilot_cuda_unavailable":
            raise
    else:
        raise RuntimeError("CUDA selection must fail without GPU access.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder, decoder = build_models(0)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()], lr=1e-4
    )

    def step() -> None:
        """Exercise one real dense-model update on a tiny CPU-only tensor."""
        optimizer.zero_grad(set_to_none=True)
        cover = torch.rand(1, 3, 32, 32)
        payload = torch.randint(0, 2, (1, 1, 32, 32)).float()
        output = quantize_image(encoder(cover, payload), straight_through=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            decoder(output), payload
        )
        torch.autograd.backward(loss)
        optimizer.step()

    step()
    saved = copy.deepcopy(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "random": random_state(),
        }
    )
    step()
    expected = copy.deepcopy((encoder.state_dict(), decoder.state_dict()))
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        torch.save(saved, directory / "state.pt")
        restored = torch.load(directory / "state.pt", weights_only=True)
        encoder.load_state_dict(restored["encoder"])
        decoder.load_state_dict(restored["decoder"])
        optimizer.load_state_dict(restored["optimizer"])
        restore_random_state(restored["random"])
        step()
        for model, state in zip((encoder, decoder), expected, strict=True):
            for name, actual in model.state_dict().items():
                torch.testing.assert_close(actual, state[name], rtol=0, atol=0)
        encoder.eval()
        inputs = (torch.zeros(1, 3, 32, 32), torch.zeros(1, 1, 32, 32))
        exported = torch.export.export(encoder, inputs)
        torch.export.save(exported, directory / "encoder.pt2")
        loaded = torch.export.load(directory / "encoder.pt2").module()
        torch.testing.assert_close(loaded(*inputs), encoder(*inputs), rtol=0, atol=0)
    print("Native CUDA image: dependencies and CPU checks passed; GPU checks pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
