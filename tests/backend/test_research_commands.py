"""Exit codes, safe diagnostics and run records of the three research commands."""

import io
import json
import signal
from collections.abc import Callable
from pathlib import Path

import pytest
from near_duplicate_fixtures import smooth_image, write_revision

from backend_service.command_line import main
from schemas.near_duplicate_audit import NearDuplicateAuditReport

VALIDATION_IDENTIFIERS = [100, 101, 102]
TRAINING_IDENTIFIERS = list(range(1000, 1012))
ARCHIVE = {
    "repository": "fixture/coco-mirror",
    "revision": "a" * 40,
    "archive_path": "val2017.zip",
    "archive_sha256": "b" * 64,
    "archive_bytes": 1,
}
ANNOTATION_FILES = {
    "captions_val2017.json": VALIDATION_IDENTIFIERS,
    "instances_val2017.json": VALIDATION_IDENTIFIERS,
    "captions_train2017.json": TRAINING_IDENTIFIERS,
}


def write_annotations(directory: Path) -> Path:
    """Write three COCO-style annotation files with images arrays."""
    directory.mkdir(parents=True)
    for name, identifiers in ANNOTATION_FILES.items():
        images = [
            {
                "id": number,
                "file_name": f"{number:012d}.jpg",
                "width": 640,
                "height": 480,
            }
            for number in identifiers
        ]
        (directory / name).write_text(json.dumps({"images": images}), encoding="utf-8")
    return directory


def freeze_document(annotations: Path, output: Path) -> dict[str, object]:
    """Describe a small freeze with four training-source and two reserve members."""
    return {
        "annotations_directory": str(annotations),
        "output_directory": str(output),
        "source_archives": [ARCHIVE],
        "seed": 3,
        "training_source_count": 4,
        "reserve_count": 2,
        "expect_official_counts": False,
    }


@pytest.fixture
def logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send run logs to the test folder and return that folder."""
    directory = tmp_path / "logs"
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(directory))
    return directory


def run_folders(logs: Path, label: str) -> list[Path]:
    """List the run folders created under one command label."""
    return sorted(path for path in logs.iterdir() if f"_{label}_" in path.name)


def events(logs: Path, label: str) -> str:
    """Join the event logs of every run under one command label."""
    return "".join(
        (folder / "events.jsonl").read_text() for folder in run_folders(logs, label)
    )


def test_freeze_and_validate_benchmark_exit_codes(
    tmp_path: Path, logs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Success prints the report; refusals return one without echoing paths."""
    annotations = write_annotations(tmp_path / "annotations")
    output = tmp_path / "frozen"
    request = tmp_path / "freeze.json"
    request.write_text(json.dumps(freeze_document(annotations, output)))
    assert main(["freeze_benchmark", str(request)]) == 0
    header = json.loads(capsys.readouterr().out)
    assert (header["member_count"], header["pilot_ready"]) == (9, False)
    assert main(["freeze_benchmark", str(request)]) == 1
    error = capsys.readouterr().err
    assert "already exists" in error and str(tmp_path) not in error
    assert main(["validate_benchmark", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["integrity"] == "verified"
    with_annotations = ["--annotations-directory", str(annotations)]
    assert main(["validate_benchmark", str(output), *with_annotations]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["integrity"] == "verified_with_annotations"
    assert report["identities_checksum"] == header["identities_checksum"]
    assert main(["validate_benchmark", str(tmp_path / "absent")]) == 1
    assert str(tmp_path) not in capsys.readouterr().err
    other = write_annotations(tmp_path / "other")
    (other / "captions_train2017.json").write_text('{"images": []}', encoding="utf-8")
    mismatch = ["--annotations-directory", str(other)]
    assert main(["validate_benchmark", str(output), *mismatch]) == 1
    assert str(tmp_path) not in capsys.readouterr().err
    assert len(run_folders(logs, "freeze_benchmark")) == 2
    assert len(run_folders(logs, "validate_benchmark")) == 4
    frozen = events(logs, "freeze_benchmark")
    assert "benchmark_identities_frozen" in frozen
    assert "research_command_completed" in frozen
    assert "benchmark_identities_verified" in events(logs, "validate_benchmark")


def test_invalid_requests_and_arguments_return_two(
    tmp_path: Path, logs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unreadable or invalid requests exit with two before any run log exists."""
    request = tmp_path / "request.json"
    request.write_text('{"annotations_directory": "PRIVATE_PATH", "unknown": true}')
    assert main(["freeze_benchmark", str(request)]) == 2
    assert "PRIVATE_PATH" not in capsys.readouterr().err
    request.write_text("not json")
    assert main(["audit_near_duplicates", str(request)]) == 2
    assert main(["audit_near_duplicates", str(tmp_path / "missing.json")]) == 2
    assert str(tmp_path) not in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_status:
        main(["validate_benchmark"])
    assert exit_status.value.code == 2
    assert not logs.exists()


def test_audit_command_writes_report_and_run_record(
    tmp_path: Path,
    logs: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A standard-input request audits the revision and records the run."""
    revision = write_revision(
        tmp_path / "datasets" / "audit_fixture" / "staging",
        [
            ("training/000.png", smooth_image(0), "training"),
            ("validation/001.png", smooth_image(1), "validation"),
        ],
    )
    document = {
        "dataset_directory": str(revision),
        "output_root": str(tmp_path / "out"),
    }
    monkeypatch.setattr(
        "sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(document).encode()))
    )
    assert main(["audit_near_duplicates", "-"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["hashed_images"], report["pilot_ready"]) == (2, False)
    (folder,) = run_folders(logs, "audit_near_duplicates")
    assert sorted(path.name for path in folder.iterdir()) == [
        "events.jsonl",
        "near_duplicate_audit_run.json",
    ]
    run = json.loads((folder / "near_duplicate_audit_run.json").read_text())
    assert run["audit_identifier"] == report["audit_identifier"]
    assert (run["hashed_images"], run["limited"]) == (2, False)
    assert run["elapsed_seconds"] >= 0 and run["peak_process_mebibytes"] > 0
    assert run["cross_split_totals"] == {"0": 0, "4": 0, "8": 0, "12": 0}
    assert str(tmp_path) not in json.dumps(run)
    lines = (folder / "events.jsonl").read_text().splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events[0] == "research_command_started"
    assert events[-1] == "research_command_completed"
    assert "near_duplicate_audit_completed" in events
    audits = tmp_path / "out" / ".audits" / "audit_fixture" / report["audit_identifier"]
    assert (audits / "near_duplicate_audit.json").is_file()
    request = tmp_path / "audit.json"
    request.write_text(json.dumps({"dataset_directory": str(tmp_path / "missing")}))
    assert main(["audit_near_duplicates", str(request)]) == 1
    error = capsys.readouterr().err
    assert "Validate the dataset" in error and str(tmp_path) not in error


def test_invalid_report_during_execution_returns_one(
    tmp_path: Path,
    logs: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A report the schema rejects is a command failure, not a request error."""

    def broken(*arguments: object, **keywords: object) -> NearDuplicateAuditReport:
        return NearDuplicateAuditReport.model_validate({})

    monkeypatch.setattr(
        "backend_service.near_duplicate_audit.audit_near_duplicates", broken
    )
    request = tmp_path / "audit.json"
    request.write_text(json.dumps({"dataset_directory": str(tmp_path)}))
    assert main(["audit_near_duplicates", str(request)]) == 1
    error = capsys.readouterr().err
    assert "invalid report" in error and "request is invalid" not in error
    assert str(tmp_path) not in error
    assert len(run_folders(logs, "audit_near_duplicates")) == 1


def raise_keyboard_interrupt(*arguments: object, **keywords: object) -> None:
    """Stand in for Ctrl-C during the audit."""
    raise KeyboardInterrupt


def raise_termination_signal(*arguments: object, **keywords: object) -> None:
    """Deliver SIGTERM to this process during the audit."""
    signal.raise_signal(signal.SIGTERM)


@pytest.mark.parametrize(
    "interruption,exit_code",
    [(raise_keyboard_interrupt, 130), (raise_termination_signal, 143)],
)
def test_interruptions_map_to_signal_exit_codes(
    tmp_path: Path,
    logs: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    interruption: Callable[..., None],
    exit_code: int,
) -> None:
    """Ctrl-C and SIGTERM return their conventional codes and restore the handler."""
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(
        "backend_service.near_duplicate_audit.audit_near_duplicates", interruption
    )
    request = tmp_path / "audit.json"
    request.write_text(json.dumps({"dataset_directory": str(tmp_path)}))
    assert main(["audit_near_duplicates", str(request)]) == exit_code
    assert signal.getsignal(signal.SIGTERM) == previous
    error = capsys.readouterr().err
    assert "before it finished" in error and str(tmp_path) not in error
    assert len(run_folders(logs, "audit_near_duplicates")) == 1
