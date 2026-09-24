"""Small dependency and usage documents bundled with independent exports."""

from typing import Literal


def requirement_bytes(dependencies: dict[str, str], device: str) -> bytes:
    """Pin runtime packages and name their explicit PyTorch wheel source."""
    index = "cu126" if device == "cuda" else "cpu"
    return (
        f"--extra-index-url https://download.pytorch.org/whl/{index}\n"
        + "\n".join(
            f"{name}=={version}" for name, version in sorted(dependencies.items())
        )
        + "\n"
    ).encode()


def usage_bytes(role: Literal["encoder", "decoder"], pilot: bool) -> bytes:
    """Explain the independent tensor and authenticated-message interfaces."""
    return (
        f"# Independent experimental {role}\n\n"
        "Use Python 3.12 and the exact requirements. CPU is the default.\n"
        + (
            "requirements-cpu.txt and requirements-cuda.txt describe "
            "separate environments.\n"
            "Use --device cuda only on the recorded CUDA host. GPU checks remain "
            "not run until the readiness report proves them. No CPU fallback occurs.\n"
            if pilot
            else "This version-one package supports CPU only.\n"
        )
        + "Only locally trusted packages may be loaded; checksums detect damage, "
        "not a malicious publisher.\n"
        "The graph uses float32, batch 1, and sides from 512 to 1024 pixels.\n"
        "Tensor command: python runtime.py tensor --image image.npy --output result.npy"
        + (" --payload payload.npy" if role == "encoder" else "")
        + "\nText commands: encode --image cover.png --output result.png, or "
        "decode --image result.png. Pass password and (for encode) message as "
        "a JSON object on stdin, never as arguments. Decode prints exact text.\n"
        "Encoder output must be checked with the matching independent decoder "
        "before the application offers it for download.\n"
        "Run python runtime.py verify to compare the graph with its public "
        "known-answer tensor at relative/absolute tolerance 0.00001.\n"
        "These packages do not establish a release-quality or secrecy claim.\n"
    ).encode()
