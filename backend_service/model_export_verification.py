"""Verify independent package execution in isolated CPU subprocesses."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

import torch
from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.model_export_io import (
    export_failure,
    read_tensor,
    verify_package,
    write_tensor,
)
from backend_service.model_exports import check_export_deadline
from backend_service.protocol_failures import RECOVERY_MESSAGE, recovery_failure
from schemas.model_exports import ExportVerification

EXPORT_SHAPES = ((512, 512), (513, 517), (1024, 1024))


def run_export_process(
    package: Path,
    arguments: list[str],
    *,
    directory: Path,
    deadline: float | None = None,
    secret_input: bytes | None = None,
) -> bytes:
    """Run a bundled entry point with no repository path or inherited Python path."""
    check_export_deadline(deadline)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    bootstrap = (
        "import runpy,sys; package=sys.argv.pop(1); "
        "sys.path.insert(0,package); "
        "runpy.run_path(package+'/runtime.py',run_name='__main__')"
    )
    timeout = 120.0 if deadline is None else min(120.0, deadline - time.monotonic())
    if timeout <= 0:
        check_export_deadline(deadline)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                bootstrap,
                str(package.resolve()),
                *arguments,
            ],
            cwd=directory,
            env=environment,
            input=secret_input,
            capture_output=True,
            check=False,
            timeout=max(0.001, timeout),
        )
    except subprocess.TimeoutExpired:
        raise ApplicationFailure(
            "model_export_budget_expired",
            "The independent model check reached its time limit.",
            422,
        ) from None
    if result.returncode:
        if result.stderr == (RECOVERY_MESSAGE + "\n").encode("utf-8"):
            raise recovery_failure()
        raise export_failure()
    check_export_deadline(deadline)
    return result.stdout


def verify_exports(
    encoder: nn.Module,
    decoder: nn.Module,
    directory: Path,
    *,
    deadline: float | None = None,
) -> ExportVerification:
    """Compare both isolated graphs to eager models at boundaries and odd sizes."""
    encoder_manifest = verify_package(directory / "encoder")
    decoder_manifest = verify_package(directory / "decoder")
    if (
        encoder_manifest.role != "encoder"
        or decoder_manifest.role != "decoder"
        or encoder_manifest.compatibility_identifier
        != decoder_manifest.compatibility_identifier
    ):
        raise export_failure()
    generator = torch.Generator(device="cpu").manual_seed(913)
    difference = 0.0
    cases = 0
    training_modes = (encoder.training, decoder.training)
    encoder.eval()
    decoder.eval()
    try:
        with tempfile.TemporaryDirectory(
            prefix=".stegolab-export-check-", dir=directory.parent.resolve()
        ) as name:
            scratch = Path(name)
            for height, width in EXPORT_SHAPES:
                check_export_deadline(deadline)
                image = torch.rand((1, 3, height, width), generator=generator)
                payload = torch.randint(
                    0, 2, (1, 1, height, width), generator=generator
                ).float()
                write_tensor(scratch / "image.npy", image)
                write_tensor(scratch / "payload.npy", payload)
                for role, model in (("encoder", encoder), ("decoder", decoder)):
                    arguments = [
                        "tensor",
                        "--image",
                        str(scratch / "image.npy"),
                        "--output",
                        str(scratch / "output.npy"),
                    ]
                    inputs: tuple[torch.Tensor, ...] = (image,)
                    if role == "encoder":
                        arguments += ["--payload", str(scratch / "payload.npy")]
                        inputs = (image, payload)
                    with torch.inference_mode():
                        expected = cast(torch.Tensor, model(*inputs))
                    run_export_process(
                        directory / role,
                        arguments,
                        directory=scratch,
                        deadline=deadline,
                    )
                    actual = read_tensor(scratch / "output.npy", expected.shape[1])
                    if not torch.allclose(expected, actual, rtol=1e-5, atol=1e-5):
                        raise export_failure()
                    difference = max(
                        difference, float(torch.max(torch.abs(expected - actual)))
                    )
                    cases += 1
                    (scratch / "output.npy").unlink()
                    del expected, actual
                (scratch / "image.npy").unlink()
                (scratch / "payload.npy").unlink()
        return ExportVerification(
            compatibility_identifier=encoder_manifest.compatibility_identifier,
            cases_checked=cases,
            maximum_absolute_difference=difference,
        )
    finally:
        encoder.train(training_modes[0])
        decoder.train(training_modes[1])
