"""Exercise independent exports, corrupt files, shapes, and atomic failures."""

import json
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from backend_service.failures import ApplicationFailure
from backend_service.model_export_io import verify_package
from backend_service.model_export_runtime import load_export
from backend_service.model_export_verification import run_export_process
from backend_service.model_exports import _package, export_models
from backend_service.model_fixture_channel import (
    IntegerChannelDecoder,
    IntegerChannelEncoder,
)
from backend_service.model_networks import (
    DenseDecoder,
    DenseEncoder,
    build_models,
    model_pair_identifier,
)
from schemas.model_exports import ExportMetadata, ExportVerification


@pytest.fixture(scope="module")
def exported_pair(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[DenseEncoder, DenseDecoder, Path]]:
    """Export one real dense pair and restore the process thread setting."""
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    encoder, decoder = build_models(411)
    destination = tmp_path_factory.mktemp("independent-models") / "pair"
    export_models(
        encoder,
        decoder,
        destination,
        ExportMetadata(
            compatibility_identifier=model_pair_identifier(encoder, decoder),
            source_identifier="public_test_fixture",
        ),
    )
    try:
        yield encoder, decoder, destination
    finally:
        torch.set_num_threads(original_threads)


def test_dense_exports_work_in_independent_processes(
    exported_pair: tuple[DenseEncoder, DenseDecoder, Path],
) -> None:
    """Prove both graphs match eager inference at all three proof dimensions."""
    encoder, decoder, directory = exported_pair
    report = ExportVerification.model_validate_json(
        (directory / "verification.json").read_bytes()
    )
    assert report.cases_checked == 6
    assert report.maximum_absolute_difference <= 1e-5
    assert encoder.training and decoder.training
    for role in ("encoder", "decoder"):
        manifest = verify_package(directory / role)
        assert manifest.role == role
        assert manifest.compatibility_identifier == report.compatibility_identifier
        assert "model.pt2" in manifest.files
        assert "known_answer.npy" in manifest.files
        run_export_process(directory / role, ["verify"], directory=directory.parent)
        assert not any(
            "training" in name or "model_networks" in name for name in manifest.files
        )
        assert {"torch", "numpy", "Pillow", "PyNaCl", "pydantic", "reedsolo"} <= set(
            manifest.dependencies
        )


def test_export_rejects_unsupported_shapes_and_wrong_role_inputs(
    exported_pair: tuple[DenseEncoder, DenseDecoder, Path],
) -> None:
    """Keep tensor batch, channels, spatial bounds, and matching sizes fixed."""
    _, _, directory = exported_pair
    _, graph = load_export(directory / "encoder")
    for shape in ((1, 3, 511, 512), (1, 3, 1025, 512), (2, 3, 512, 512)):
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            graph(torch.zeros(shape), torch.zeros(shape[0], 1, shape[2], shape[3]))
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        graph(torch.zeros(1, 3, 512, 512), torch.zeros(1, 1, 513, 512))


def test_export_checksums_prevent_loading_changed_graph(
    exported_pair: tuple[DenseEncoder, DenseDecoder, Path],
    tmp_path: Path,
) -> None:
    """Reject corrupted graph bytes before invoking the export deserializer."""
    _, _, directory = exported_pair
    package = tmp_path / "encoder"
    shutil.copytree(directory / "encoder", package)
    graph = package / "model.pt2"
    content = bytearray(graph.read_bytes())
    content[len(content) // 2] ^= 1
    graph.write_bytes(content)
    with pytest.raises(ApplicationFailure, match="missing, damaged, or incompatible"):
        load_export(package)


def test_export_preserves_existing_directory_and_expired_budget(
    exported_pair: tuple[DenseEncoder, DenseDecoder, Path],
    tmp_path: Path,
) -> None:
    """Keep existing files and never publish a package after budget expiry."""
    encoder, decoder, _ = exported_pair
    metadata = ExportMetadata(
        compatibility_identifier=model_pair_identifier(encoder, decoder),
        source_identifier="public_test_fixture",
    )
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_bytes(b"unchanged")
    with pytest.raises(ApplicationFailure):
        export_models(encoder, decoder, existing, metadata)
    assert (existing / "keep").read_bytes() == b"unchanged"
    with pytest.raises(ApplicationFailure) as failure:
        export_models(
            encoder,
            decoder,
            tmp_path / "expired",
            metadata,
            deadline=time.monotonic() - 1,
        )
    assert failure.value.code == "model_export_budget_expired"
    assert not (tmp_path / "expired").exists()
    assert not list(tmp_path.glob(".stegolab-export-*"))


def test_failed_export_cleans_partial_stage(
    exported_pair: tuple[DenseEncoder, DenseDecoder, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A storage failure leaves no published directory or private staging."""
    encoder, decoder, _ = exported_pair

    def fail_write(*arguments: object) -> None:
        """Simulate full storage during package creation."""
        raise OSError("test storage failure")

    monkeypatch.setattr("backend_service.model_exports._write_file", fail_write)
    with pytest.raises(ApplicationFailure):
        export_models(
            encoder,
            decoder,
            tmp_path / "failed",
            ExportMetadata(
                compatibility_identifier=model_pair_identifier(encoder, decoder),
                source_identifier="public_test_fixture",
            ),
        )
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".stegolab-export-*"))
    assert encoder.training and decoder.training


def test_failed_parity_never_publishes_export(
    exported_pair: tuple[DenseEncoder, DenseDecoder, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a completed staged pair when independent verification fails."""
    encoder, decoder, _ = exported_pair
    destination = tmp_path / "failed_parity"

    def fail_verification(
        first: nn.Module,
        second: nn.Module,
        directory: Path,
        *,
        deadline: float | None = None,
    ) -> ExportVerification:
        """Fail only after both staged packages exist and before publication."""
        assert (directory / "encoder" / "model.pt2").is_file()
        assert (directory / "decoder" / "model.pt2").is_file()
        assert not destination.exists()
        raise ApplicationFailure("model_export_unavailable", "Public parity failure.")

    monkeypatch.setattr(
        "backend_service.model_export_verification.verify_exports", fail_verification
    )
    with pytest.raises(ApplicationFailure, match="Public parity failure"):
        export_models(
            encoder,
            decoder,
            destination,
            ExportMetadata(
                compatibility_identifier=model_pair_identifier(encoder, decoder),
                source_identifier="public_test_fixture",
            ),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".stegolab-export-*"))


@pytest.mark.parametrize("total_gib", [50, 200])
def test_export_keeps_disk_reserve(
    exported_pair: tuple[DenseEncoder, DenseDecoder, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    total_gib: int,
) -> None:
    """Refuse exports whose maximum pending writes cross either reserve rule."""
    from collections import namedtuple

    Usage = namedtuple("Usage", "total used free")
    total = total_gib * 1024**3
    free = max(10 * 1024**3, total // 10) + 128 * 1024**2
    monkeypatch.setattr(
        "backend_service.model_export_io.shutil.disk_usage",
        lambda _: Usage(total, total - free, free),
    )
    encoder, decoder, _ = exported_pair
    with pytest.raises(ApplicationFailure) as failure:
        export_models(
            encoder,
            decoder,
            tmp_path / "low_space",
            ExportMetadata(
                compatibility_identifier=model_pair_identifier(encoder, decoder),
                source_identifier="public_test_fixture",
            ),
        )
    assert failure.value.code == "model_export_space"
    assert not (tmp_path / "low_space").exists()
    assert not list(tmp_path.glob(".stegolab-export-*"))


def test_standalone_text_wrappers_preserve_authenticated_utf8(tmp_path: Path) -> None:
    """Check real protocol and saved PNG through separate public fixture graphs."""
    encoder, decoder = IntegerChannelEncoder(), IntegerChannelDecoder()
    metadata = ExportMetadata(
        compatibility_identifier=model_pair_identifier(encoder, decoder),
        source_identifier="public_integer_wrapper_fixture_not_a_neural_model",
    )
    _package(encoder, tmp_path / "encoder", "encoder", metadata, None)
    _package(decoder, tmp_path / "decoder", "decoder", metadata, None)
    image = tmp_path / "cover.png"
    Image.new("RGB", (513, 517), (100, 121, 140)).save(image)
    message = "  שלום, café e\u0301 🌍\n\n"
    encoded = tmp_path / "stego.png"
    run_export_process(
        tmp_path / "encoder",
        ["encode", "--image", str(image), "--output", str(encoded)],
        directory=tmp_path,
        secret_input=json.dumps({"message": message, "password": "public"}).encode(),
    )
    decoded = run_export_process(
        tmp_path / "decoder",
        ["decode", "--image", str(encoded)],
        directory=tmp_path,
        secret_input=b'{"password":"public"}',
    )
    assert decoded.decode() == message
    with pytest.raises(ApplicationFailure):
        run_export_process(
            tmp_path / "decoder",
            ["decode", "--image", str(encoded)],
            directory=tmp_path,
            secret_input=b'{"password":"wrong"}',
        )
