"""Worker boundaries tested without starting an actual model experiment."""

import fcntl
import os
import signal
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, cast

import pytest

from backend_service import workflow_execution
from backend_service.workflow_lock import proof_busy
from schemas.training import TrainingRun, TrainingStep


class SavingProcess:
    """Model startup followed by the first safe optimizer boundary."""

    def __init__(self, root: Path, output: BinaryIO) -> None:
        """Keep the fake child disconnected from all real worker processes."""
        self.root = root
        self.output = output
        self.stdin = BytesIO()
        self.returncode: int | None = None
        self.polls = 0
        self.signals: list[int] = []

    def poll(self) -> int | None:
        """Publish the first step only after the startup gap was observed."""
        self.polls += 1
        if self.polls == 2:
            assert not self.signals
            path = self.root / "logs" / "date_test_attempt" / "steps.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(
                TrainingStep(
                    global_step=1,
                    bit_loss=0.5,
                    image_loss=0.1,
                    image_loss_weight=0.0,
                    elapsed_seconds=1.0,
                ).model_dump_json()
                + "\n"
            )
        return self.returncode

    def send_signal(self, number: int) -> None:
        """Finish only after returning a structured safe-save result."""
        self.signals.append(number)
        self.output.write(
            TrainingRun(
                experiment_identifier="test",
                status="stopped",
                global_step=1,
                checkpoint="checkpoints/test/checkpoint",
                dataset_revision="revision",
                compatibility_identifier="model",
                elapsed_seconds=1.0,
                remaining_experiment_seconds=6000.0,
                selected_training_identities=[],
                selected_tuning_identities=[],
                stop_signal=15,
            )
            .model_dump_json()
            .encode()
        )
        self.output.flush()
        self.returncode = 143


def test_stop_waits_for_a_safe_boundary_and_keeps_saved_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop during startup never kills the trainer before its handler exists."""
    processes: list[SavingProcess] = []

    def spawn(*arguments: object, **options: object) -> SavingProcess:
        """Substitute a bounded child while exercising the real supervisor loop."""
        environment = cast(dict[str, str], options["env"])
        assert environment["STEGOLAB_WORKER_PARENT_IDENTIFIER"] == str(os.getpid())
        child = SavingProcess(tmp_path, cast(BinaryIO, options["stdout"]))
        processes.append(child)
        return child

    monkeypatch.setattr("backend_service.workflow_execution.subprocess.Popen", spawn)
    monkeypatch.setattr("backend_service.workflow_execution.time.sleep", lambda _: None)
    progress: list[TrainingStep] = []
    code, result = workflow_execution.execute_command(
        tmp_path,
        "train",
        {"experiment_identifier": "test"},
        on_process=lambda _: None,
        on_progress=progress.append,
        stop_requested=lambda: True,
    )
    assert code == 143
    assert result["status"] == "stopped"
    assert len(progress) == 1
    assert processes[0].signals == [signal.SIGTERM]
    assert not (tmp_path / "state").exists()


def test_external_proof_lock_is_observed_without_modification(tmp_path: Path) -> None:
    """Browser work can wait for CLI ownership without spending or repairing it."""
    assert not proof_busy(tmp_path)
    assert not (tmp_path / "state").exists()
    path = tmp_path / "state" / "cpu_proof" / ".proof.lock"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unchanged")
    with path.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert proof_busy(tmp_path)
    assert not proof_busy(tmp_path)
    assert path.read_bytes() == b"unchanged"


def test_worker_uses_expected_parent_even_after_early_parent_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reparented child never treats PID one as the intended API owner."""
    from backend_service import workflow_worker

    captured: list[int] = []

    class Watcher:
        """Capture the monitor arguments without starting a real thread."""

        def __init__(self, **options: object) -> None:
            """Record the parent supplied by the spawning API process."""
            captured.extend(cast(tuple[int], options["args"]))

        def start(self) -> None:
            """Keep this worker-launch test free of operating-system signals."""

    monkeypatch.setenv("STEGOLAB_WORKER_PARENT_IDENTIFIER", "12345")
    monkeypatch.setattr("backend_service.workflow_worker.os.getppid", lambda: 1)
    monkeypatch.setattr("backend_service.workflow_worker.threading.Thread", Watcher)
    monkeypatch.setattr(workflow_worker, "main", lambda _: 0)
    assert workflow_worker.run() == 0
    assert captured == [12345]
