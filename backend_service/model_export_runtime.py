"""Independent CPU tensor and text runtime copied into each model package."""

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Literal, Never, cast

import numpy as np
import torch
from PIL import Image
from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.image_output import write_png
from backend_service.image_preparation import _read_pixels, read_prepared_png
from backend_service.message_protocol import decode_message, encode_message
from backend_service.model_export_devices import export_device
from backend_service.model_export_examples import verify_example
from backend_service.model_export_io import (
    array_tensor,
    export_failure,
    read_export_file,
    read_tensor,
    tensor_array,
    verify_package,
    write_tensor,
)
from backend_service.payload_map import create_payload_map, recover_payload_bytes
from schemas.model_exports import ModelExportManifest
from schemas.protocol import ProtocolContext


def load_export(
    directory: Path, *, device: Literal["cpu", "cuda"] = "cpu"
) -> tuple[ModelExportManifest, nn.Module]:
    """Load a locally trusted graph after integrity and runtime checks."""
    try:
        selected_device = export_device(device)
        manifest = verify_package(directory, device=device)
        program = torch.export.load(
            BytesIO(read_export_file(directory / manifest.graph_filename))
        )
        return manifest, program.module().to(selected_device)
    except ApplicationFailure:
        raise
    except (OSError, ValueError, RuntimeError, KeyError, ImportError):
        raise export_failure() from None


def _image_tensor(image: Image.Image) -> torch.Tensor:
    """Convert prepared RGB integers to the graph's [0,1] input."""
    pixels = np.array(image.convert("RGB"), dtype=np.float32) / np.float32(255)
    return array_tensor(pixels.transpose(2, 0, 1)[None, ...], 3)


def _context(manifest: ModelExportManifest, image: Image.Image) -> ProtocolContext:
    """Bind the protocol to this model pair and supported dimensions."""
    if not all(512 <= side <= 1024 for side in image.size):
        raise export_failure()
    return ProtocolContext(
        width=image.width,
        height=image.height,
        compatibility_identifier=manifest.compatibility_identifier,
    )


def encode_png(
    directory: Path,
    image_path: Path,
    output_path: Path,
    message: str,
    password: str,
    *,
    device: Literal["cpu", "cuda"] = "cpu",
) -> None:
    """Encode to quantized PNG without requiring a decoder package."""
    manifest, graph = load_export(directory, device=device)
    if manifest.role != "encoder":
        raise export_failure()
    prepared = _read_pixels(image_path, recover=False).image
    context = _context(manifest, prepared)
    protected = encode_message(message, password, context)
    payload = create_payload_map(protected, context).astype(np.float32)[None, ...]
    with torch.inference_mode():
        result = cast(
            torch.Tensor,
            graph(
                _image_tensor(prepared).to(device), array_tensor(payload, 1).to(device)
            ),
        )
        pixels = torch.round(result.clamp(0, 1) * 255).to(torch.uint8)
    output = Image.fromarray(pixels[0].permute(1, 2, 0).cpu().numpy())
    if prepared.mode == "RGBA":
        output.putalpha(prepared.getchannel("A"))
    write_png(output, output_path)


def decode_png(
    directory: Path,
    image_path: Path,
    password: str,
    *,
    device: Literal["cpu", "cuda"] = "cpu",
) -> str:
    """Recover authenticated exact text using only the decoder package."""
    manifest, graph = load_export(directory, device=device)
    if manifest.role != "decoder":
        raise export_failure()
    image = read_prepared_png(image_path).image
    context = _context(manifest, image)
    with torch.inference_mode():
        result = cast(torch.Tensor, graph(_image_tensor(image).to(device)))
    protected = recover_payload_bytes(tensor_array(result)[0], context)
    return decode_message(protected, password, context)


def run_tensor(
    directory: Path,
    image_path: Path,
    output_path: Path,
    payload_path: Path | None,
    *,
    device: Literal["cpu", "cuda"] = "cpu",
) -> None:
    """Run one package with public NumPy tensors and no training imports."""
    manifest, graph = load_export(directory, device=device)
    image = read_tensor(image_path, 3).to(device)
    if not torch.all((image >= 0) & (image <= 1)):
        raise export_failure()
    with torch.inference_mode():
        if manifest.role == "encoder":
            if payload_path is None:
                raise export_failure()
            payload = read_tensor(payload_path, 1).to(device)
            if payload.shape[2:] != image.shape[2:] or not torch.all(
                (payload == 0) | (payload == 1)
            ):
                raise export_failure()
            result = cast(torch.Tensor, graph(image, payload))
        else:
            if payload_path is not None:
                raise export_failure()
            result = cast(torch.Tensor, graph(image))
    write_tensor(output_path, result)


def _secrets(action: Literal["encode", "decode"]) -> dict[str, str]:
    """Read bounded secret input from stdin without command-line exposure."""
    content = sys.stdin.buffer.read(16_385)
    if len(content) > 16_384:
        raise export_failure()
    value = json.loads(content)
    keys = {"password", "message"} if action == "encode" else {"password"}
    if not isinstance(value, dict) or set(value) != keys:
        raise export_failure()
    if not all(isinstance(item, str) for item in value.values()):
        raise export_failure()
    return cast(dict[str, str], value)


class SafeParser(argparse.ArgumentParser):
    """Keep mistaken secret arguments out of parser diagnostics."""

    def error(self, message: str) -> Never:
        """Replace argument names and values with fixed public guidance."""
        self.exit(
            2, "The model command arguments are invalid. Use --help for examples.\n"
        )


def main(command_arguments: list[str] | None = None) -> int:
    """Run a package directly; decode writes only authenticated user text."""
    parser = SafeParser(
        prog="stegolab-model", description="Experimental independent CPU model"
    )
    parser.add_argument("action", choices=("tensor", "encode", "decode", "verify"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(command_arguments)
    directory = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    try:
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] in ("backend_service", "schemas"):
                origin = getattr(module, "__file__", None)
                if origin is None or not Path(origin).resolve().is_relative_to(
                    directory
                ):
                    raise export_failure()
        if arguments.action == "verify":
            manifest, graph = load_export(directory, device=arguments.device)
            channels = 3 if manifest.role == "encoder" else 1
            expected = read_tensor(directory / "known_answer.npy", channels)
            verify_example(graph, manifest.role, expected, device=arguments.device)
            return 0
        if arguments.image is None:
            raise export_failure()
        if arguments.action == "tensor":
            if arguments.output is None:
                raise export_failure()
            run_tensor(
                directory,
                arguments.image,
                arguments.output,
                arguments.payload,
                device=arguments.device,
            )
        elif arguments.action == "encode":
            if arguments.output is None or arguments.payload is not None:
                raise export_failure()
            secrets = _secrets("encode")
            encode_png(
                directory,
                arguments.image,
                arguments.output,
                secrets["message"],
                secrets["password"],
                device=arguments.device,
            )
        else:
            if arguments.output is not None or arguments.payload is not None:
                raise export_failure()
            secrets = _secrets("decode")
            text = decode_png(
                directory, arguments.image, secrets["password"], device=arguments.device
            )
            # Write UTF-8 bytes directly so the caller's locale cannot alter them.
            sys.stdout.buffer.write(text.encode("utf-8"))
            sys.stdout.buffer.flush()
        return 0
    except ApplicationFailure as failure:
        print(failure.message, file=sys.stderr)
    except (OSError, ValueError, RuntimeError, KeyError, MemoryError):
        print(export_failure().message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
