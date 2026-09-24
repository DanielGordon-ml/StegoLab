"""Publish separate, checksummed experimental encoder and decoder packages."""

import hashlib
import importlib.metadata
import os
import platform
import shutil
import tempfile
import time
from collections.abc import Buffer
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from backend_service.dataset_files import directory_descriptor
from backend_service.dataset_storage_ops import (
    flush_directory_tree,
    publish_directory,
)
from backend_service.failures import ApplicationFailure
from backend_service.model_export_examples import example_output
from backend_service.model_export_io import (
    MAXIMUM_EXPORT_BYTES,
    MAXIMUM_EXPORT_FILE,
    export_failure,
    require_export_space,
    verify_package,
)
from backend_service.model_networks import model_pair_identifier
from schemas.model_exports import (
    ExportFile,
    ExportMetadata,
    ModelExportManifest,
    ModelExportSummary,
)

RUNTIME_MODULES = (
    "__init__",
    "failures",
    "image_color",
    "image_diagnostics",
    "image_output",
    "image_png_stream",
    "image_policy",
    "image_preparation",
    "image_validation",
    "message_correction",
    "message_frame",
    "message_protocol",
    "model_export_examples",
    "model_export_io",
    "payload_capacity",
    "payload_map",
    "protocol_failures",
)
RUNTIME_SCHEMAS = (
    "__init__",
    "base",
    "image_dimensions",
    "images",
    "model_exports",
    "protocol",
)
DEPENDENCIES = (
    "torch",
    "numpy",
    "Pillow",
    "PyNaCl",
    "reedsolo",
    "pydantic",
    "cffi",
    "pycparser",
    "pydantic_core",
    "typing_extensions",
    "typing-inspection",
    "annotated-types",
    "filelock",
    "sympy",
    "mpmath",
    "networkx",
    "Jinja2",
    "MarkupSafe",
    "fsspec",
)


class _BoundedExportBuffer(BytesIO):
    """Limit graph serialization before it can consume unbounded storage."""

    def write(self, buffer: Buffer, /) -> int:
        """Reject writes beyond the per-file package limit."""
        if self.tell() + memoryview(buffer).nbytes > MAXIMUM_EXPORT_FILE:
            raise export_failure()
        return super().write(buffer)


def check_export_deadline(deadline: float | None) -> None:
    """Stop between bounded export operations when the proof budget expires."""
    if deadline is not None and time.monotonic() >= deadline:
        raise ApplicationFailure(
            "model_export_budget_expired",
            "The export time budget ended. No incomplete package was published.",
            422,
        )


def _write_file(path: Path, content: bytes) -> ExportFile:
    """Save and flush one bounded package file with its checksum."""
    if not 0 < len(content) <= MAXIMUM_EXPORT_FILE:
        raise export_failure()
    require_export_space(path.parent, len(content))
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    return ExportFile(
        checksum=hashlib.sha256(content).hexdigest(), size_bytes=len(content)
    )


def _runtime_files() -> dict[str, bytes]:
    """Collect only audited inference, image, and protocol modules."""
    root = Path(__file__).resolve().parent.parent
    result = {
        "runtime.py": (root / "backend_service/model_export_runtime.py").read_bytes()
    }
    for package, names in (
        ("backend_service", RUNTIME_MODULES),
        ("schemas", RUNTIME_SCHEMAS),
    ):
        for name in names:
            relative = f"{package}/{name}.py"
            result[relative] = (root / relative).read_bytes()
    return result


def _dependencies() -> dict[str, str]:
    """Pin the independent runtime and its shared CPU transitive packages."""
    versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    versions["torch"] = versions["torch"].split("+", 1)[0]
    return versions


def _graph_bytes(model: nn.Module, role: Literal["encoder", "decoder"]) -> bytes:
    """Export batch-one CPU inference with shared bounded spatial dimensions."""
    if any(value.device.type != "cpu" for value in model.state_dict().values()):
        raise export_failure()
    model.eval()
    height = torch.export.Dim("height", min=512, max=1024)
    width = torch.export.Dim("width", min=512, max=1024)
    shape = {2: height, 3: width}
    cover = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
    inputs: tuple[torch.Tensor, ...] = (cover,)
    dynamic_shapes: tuple[dict[int, torch.export.Dim], ...] = (shape,)
    if role == "encoder":
        inputs = (cover, torch.zeros(1, 1, 512, 512, dtype=torch.float32))
        dynamic_shapes = (shape, shape)
    with torch.no_grad():
        program = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
    with _BoundedExportBuffer() as output:
        torch.export.save(program, output)
        return output.getvalue()


def _package(
    model: nn.Module,
    directory: Path,
    role: Literal["encoder", "decoder"],
    metadata: ExportMetadata,
    deadline: float | None,
) -> None:
    """Assemble and verify a complete package inside private staging."""
    directory.mkdir()
    (directory / "backend_service").mkdir()
    (directory / "schemas").mkdir()
    check_export_deadline(deadline)
    files = _runtime_files()
    files["model.pt2"] = _graph_bytes(model, role)
    files["known_answer.npy"] = example_output(model, role)
    check_export_deadline(deadline)
    dependencies = _dependencies()
    files["requirements.txt"] = (
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        + "\n".join(
            f"{name}=={version}" for name, version in sorted(dependencies.items())
        )
        + "\n"
    ).encode()
    files["README.md"] = (
        f"# Independent experimental {role}\n\n"
        "Use Python 3.12 and the exact requirements. Run on CPU.\n"
        "Only locally trusted packages may be loaded; checksums detect damage, "
        "not a malicious publisher.\n"
        "The graph uses float32, batch 1, and sides from 512 to 1024 pixels.\n"
        "Tensor command: python runtime.py tensor --image image.npy "
        "--output result.npy"
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
    if sum(len(content) for content in files.values()) > MAXIMUM_EXPORT_BYTES:
        raise export_failure()
    records = {
        name: _write_file(directory / name, content)
        for name, content in sorted(files.items())
    }
    manifest = ModelExportManifest(
        **metadata.model_dump(),
        role=role,
        files=records,
        dependencies=dependencies,
        producer_environment={
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "system": platform.system(),
            "architecture": platform.machine(),
        },
    )
    _write_file(
        directory / "manifest.json", manifest.model_dump_json(indent=2).encode()
    )
    verify_package(directory)


def export_models(
    encoder: nn.Module,
    decoder: nn.Module,
    destination: Path,
    metadata: ExportMetadata,
    *,
    deadline: float | None = None,
) -> ModelExportSummary:
    """Publish a verified independent pair atomically without overwriting files."""
    from backend_service.model_export_verification import verify_exports

    stage: Path | None = None
    training_modes = (encoder.training, decoder.training)
    try:
        metadata = ExportMetadata.model_validate(metadata)
        check_export_deadline(deadline)
        destination = destination.parent.resolve(strict=True) / destination.name
        if (
            destination.exists()
            or destination.is_symlink()
            or metadata.compatibility_identifier
            != model_pair_identifier(encoder, decoder)
        ):
            raise export_failure()
        require_export_space(destination.parent, 3 * MAXIMUM_EXPORT_BYTES)
        stage = Path(
            tempfile.mkdtemp(prefix=".stegolab-export-", dir=destination.parent)
        )
        _package(encoder, stage / "encoder", "encoder", metadata, deadline)
        _package(decoder, stage / "decoder", "decoder", metadata, deadline)
        verification = verify_exports(encoder, decoder, stage, deadline=deadline)
        _write_file(
            stage / "verification.json", verification.model_dump_json(indent=2).encode()
        )
        check_export_deadline(deadline)
        flush_directory_tree(stage)
        publish_directory(stage, destination)
        stage = None
        with directory_descriptor(destination.parent) as descriptor:
            os.fsync(descriptor)
        return ModelExportSummary(
            **metadata.model_dump(),
            directory=str(destination),
            encoder_directory=str(destination / "encoder"),
            decoder_directory=str(destination / "decoder"),
        )
    except ApplicationFailure as failure:
        if failure.code.startswith("dataset_"):
            raise export_failure() from None
        raise
    except (OSError, ValueError, RuntimeError, ImportError):
        raise export_failure() from None
    finally:
        encoder.train(training_modes[0])
        decoder.train(training_modes[1])
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
