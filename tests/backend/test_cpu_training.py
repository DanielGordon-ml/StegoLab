"""Exercise real optimizer continuation and the bounded experimental command path."""

import json
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from pydantic import ValidationError

from backend_service.command_line import main
from backend_service.cpu_training import TrainingSession, create_session, training_step
from backend_service.model_data import ProofData
from backend_service.training_checkpoints import save_checkpoint
from schemas.datasets import DatasetManifest
from schemas.training import TrainingConfiguration, TrainingRequest, TrainingRun


@pytest.fixture
def proof_data(monkeypatch: pytest.MonkeyPatch) -> ProofData:
    """Isolate the trainer from dataset I/O already covered by reader tests."""
    crops = tuple(torch.full((3, 256, 256), 0.2 + index / 10) for index in range(4))
    data = ProofData(
        training_crops=crops,
        tuning_crops=crops,
        manifest=cast(DatasetManifest, SimpleNamespace(revision="a" * 64)),
        identities={"train": ("a", "b", "c", "d"), "tuning": ("e", "f", "g", "h")},
        tuning_examples=(),
    )
    monkeypatch.setattr(
        "backend_service.cpu_training.load_proof_data",
        lambda *arguments, **keywords: data,
    )
    return data


def request(root: Path) -> TrainingRequest:
    """Use one CPU thread and the same fixed learning profile in every check."""
    return TrainingRequest(
        experiment_identifier="test_run",
        output_root=str(root),
        dataset_directory=str(root / "dataset"),
        configuration=TrainingConfiguration(cpu_threads=1),
        stop_after_step=1,
    )


def test_dense_optimizer_resume_is_exact(tmp_path: Path, proof_data: ProofData) -> None:
    """Two real dense-model updates match an interrupted and restored pair."""
    original = request(tmp_path)
    continuous = create_session(original)
    training_step(continuous)
    expected_loss = training_step(continuous)
    expected_random = torch.rand(8)
    interrupted = create_session(original)
    training_step(interrupted)
    summary = save_checkpoint(
        tmp_path / "checkpoints",
        interrupted.encoder,
        interrupted.decoder,
        interrupted.optimizer,
        global_step=1,
        sampler_state={"next_cover": 0},
        configuration=original.configuration.model_dump(mode="json"),
        identities=interrupted.identities,
    )
    resumed = create_session(
        original.model_copy(
            update={
                "resume_checkpoint": str(
                    tmp_path / "checkpoints" / summary.checkpoint_identifier
                )
            }
        )
    )
    assert resumed.global_step == 1
    assert training_step(resumed) == expected_loss
    for expected, actual in (
        (continuous.encoder, resumed.encoder),
        (continuous.decoder, resumed.decoder),
    ):
        for name, values in expected.state_dict().items():
            assert torch.equal(values, actual.state_dict()[name]), name
    assert torch.equal(torch.rand(8), expected_random)


def test_cli_train_and_resume_keep_one_budget(
    tmp_path: Path,
    proof_data: ProofData,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run real training through JSON CLI and retain accounting across resume."""
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))
    document = tmp_path / "request.json"
    original = request(tmp_path)
    document.write_text(original.model_dump_json())
    assert main(["train", str(document)]) == 0
    result = TrainingRun.model_validate_json(capsys.readouterr().out)
    assert result.global_step == 1
    assert result.learning_gate == "not_measured"
    ledger_path = tmp_path / "state" / "cpu_proof" / "ledger.json"
    first = json.loads(ledger_path.read_text())
    resume = original.model_copy(
        update={
            "resume_checkpoint": str(tmp_path / result.checkpoint),
            "stop_after_step": 2,
        }
    )
    document.write_text(resume.model_dump_json())
    assert main(["train", str(document)]) == 0
    assert TrainingRun.model_validate_json(capsys.readouterr().out).global_step == 2
    ledger = json.loads(ledger_path.read_text())
    assert len(ledger["experiments"]) == 1
    assert ledger["consumed_seconds"] > first["consumed_seconds"] > 0


def test_cli_rejects_private_invalid_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject unexpected secret inputs without echoing them to output or logs."""
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))
    document = tmp_path / "request.json"
    document.write_text('{"password":"private-model-sentinel"}')
    assert main(["train", str(document)]) == 2
    captured = capsys.readouterr()
    assert "private-model-sentinel" not in captured.out + captured.err
    for path in (tmp_path / "logs").rglob("*.jsonl"):
        assert "private-model-sentinel" not in path.read_text()


def test_fixed_training_settings_reject_type_lookalikes(tmp_path: Path) -> None:
    """Reject permissive Literal comparisons at the public request boundary."""
    for value in (True, 256.0, "256"):
        with pytest.raises(ValidationError):
            TrainingConfiguration.model_validate({"crop_size": value})
    values = request(tmp_path).model_dump(mode="json")
    for version in (True, 1.0, "1"):
        values["schema_version"] = version
        with pytest.raises(ValidationError):
            TrainingRequest.model_validate(values)


def test_termination_finishes_step_saves_and_restores_handlers(
    tmp_path: Path,
    proof_data: ProofData,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A signal inside an active update saves its completed state before exiting."""
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted_step(session: TrainingSession) -> tuple[float, float, float]:
        """Deliver a real termination signal before finishing the first update."""
        signal.raise_signal(signal.SIGTERM)
        return training_step(session)

    monkeypatch.setattr("backend_service.cpu_training.training_step", interrupted_step)
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))
    document = tmp_path / "request.json"
    document.write_text(
        request(tmp_path).model_copy(update={"stop_after_step": 2}).model_dump_json()
    )
    assert main(["train", str(document)]) == 143
    report = TrainingRun.model_validate_json(capsys.readouterr().out)
    assert report.status == "stopped" and report.global_step == 1
    assert (tmp_path / report.checkpoint / "metadata.json").is_file()
    assert signal.getsignal(signal.SIGTERM) == previous
