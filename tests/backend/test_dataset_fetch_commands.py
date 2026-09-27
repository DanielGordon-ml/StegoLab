"""The fetch_dataset and inspect_source commands: exit codes, records, signals."""

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from dataset_source_fixtures import LoopbackServer
from fetch_fixtures import (
    FAKE_TOKEN,
    Workspace,
    archive_routes,
    archive_spec,
    div2k_archive,
    factory_for,
    fetch_document,
    gated_spec,
    partial_files,
    staging_is_empty,
    workspace,
)
from hub_fixtures import FakeHub, gated_routes

import backend_service
from backend_service.command_line import main
from backend_service.dataset_sources import fetch_downloads
from schemas.dataset_fetch import DatasetFetchSummary
from schemas.dataset_sources import DatasetInspection

DRIVER = '''\
"""Run the real command with a transport that reaches the test's loopback server."""

import socket
import sys

from backend_service.command_line import main
from backend_service.dataset_sources import fetch_downloads
from backend_service.dataset_sources.address_policy import AddressPolicy
from backend_service.dataset_sources.transport import SecureTransport

port = int(sys.argv[2])


def connector(address, host, ignored_port, timeout):
    """Connect to the loopback server whatever port the address named."""
    return socket.create_connection((address, port), timeout=timeout)


def factory(authorization):
    """Build a transport that allows loopback and skips TLS."""
    return SecureTransport(
        AddressPolicy(allow_loopback=True),
        authorization=authorization,
        connector=connector,
        read_timeout=5.0,
    )


fetch_downloads.default_transport_factory = factory
raise SystemExit(main(["fetch_dataset", sys.argv[1]]))
'''


@pytest.fixture
def space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """Provide a workspace whose logs folder the commands write into."""
    created = workspace(tmp_path)
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(created.logs))
    for variable in ("HF_TOKEN", "HF_ACCESS_TOKEN", "HF_TOKEN_FILE"):
        monkeypatch.delenv(variable, raising=False)
    return created


def write_request(tmp_path: Path, name: str, document: Any) -> Path:
    """Save a request document as JSON and return its path."""
    path = tmp_path / name
    text = document if isinstance(document, str) else json.dumps(document)
    path.write_text(text)
    return path


def only_run(space: Workspace, label: str) -> Path:
    """Return the single run folder that carries the given label."""
    runs = sorted(space.logs.glob(f"*_{label}_*"))
    assert len(runs) == 1, runs
    return runs[0]


def test_cli_fetch_and_inspect_write_run_records_without_the_token(
    space: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command prints the summary, records the run and never logs the token."""
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    archive = div2k_archive()
    with LoopbackServer(archive_routes(archive)) as server:
        factory, _ = factory_for(server)
        monkeypatch.setattr(fetch_downloads, "default_transport_factory", factory)
        document = fetch_document(space, archive_spec(), run_label="cli_fetch")
        request = write_request(tmp_path, "fetch.json", document.model_dump_json())
        assert main(["fetch_dataset", str(request)]) == 0
        printed = capsys.readouterr().out
        summary = DatasetFetchSummary.model_validate_json(printed)
        assert summary.status == "completed" and summary.dataset is not None
        run = only_run(space, "cli_fetch")
        events = (run / "events.jsonl").read_text()
        for event in (
            "dataset_command_started",
            "fetch_resolving",
            "download_completed",
            "cache_entry_published",
            "source_materialized",
            "dataset_preparation_completed",
            "fetch_completed",
            "dataset_command_completed",
        ):
            assert f'"event": "{event}"' in events
        record = json.loads((run / "dataset_fetch_run.json").read_text())
        assert record["event"] == "dataset_fetch_completed"
        assert record["summary"]["status"] == "completed"
        assert record["elapsed_seconds"] >= 0
        progress = json.loads((run / "fetch_progress.json").read_text())
        assert progress["phase"] == "completed" and progress["files_completed"] == 4
        for text in (printed, events, json.dumps(record), json.dumps(progress)):
            assert FAKE_TOKEN not in text
        inspection_request = {
            "source": archive_spec(),
            "source_name": "div2k_sample",
            "data_root": str(space.data_root),
            "cache_root": str(space.cache_root),
        }
        stream = io.TextIOWrapper(io.BytesIO(json.dumps(inspection_request).encode()))
        monkeypatch.setattr(sys, "stdin", stream)
        assert main(["inspect_source", "-"]) == 0
    inspection = DatasetInspection.model_validate_json(capsys.readouterr().out)
    assert (inspection.raw_folder, inspection.cached_bytes) == (
        "reusable",
        len(archive),
    )
    assert inspection.download_bytes == len(archive)
    assert inspection.server_token_configured and inspection.access == "available"
    assert len(sorted(space.logs.glob("*/events.jsonl"))) >= 2


def test_cli_gated_source_exits_1_with_guidance_and_no_token_in_events(
    space: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A gated repository is refused before any download; the token stays private."""
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    hub = FakeHub(gated_routes())
    monkeypatch.setattr(
        fetch_downloads, "default_transport_factory", hub.transport_factory
    )
    document = fetch_document(
        space,
        gated_spec(),
        source_name="gated",
        dataset_name="gated",
        run_label="cli_gated",
    )
    request = write_request(tmp_path, "gated.json", document.model_dump_json())
    assert main(["fetch_dataset", str(request)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "Accept its terms" in captured.err
    assert hub.requests[0].headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
    run = only_run(space, "cli_gated")
    events = (run / "events.jsonl").read_text()
    assert '"event": "hugging_face_access_required"' in events
    assert '"event": "source_access_required"' in events
    assert FAKE_TOKEN not in events and FAKE_TOKEN not in captured.err
    assert not (run / "dataset_fetch_run.json").exists()
    assert sorted(space.cache_root.rglob("*")) == []


@pytest.mark.parametrize(
    ("command", "content"),
    [
        ("fetch_dataset", '{"run_label": "x", "source": "PRIVATE_VALUE"'),
        ("fetch_dataset", '{"run_label": "x", "unknown": "PRIVATE_VALUE"}'),
        ("inspect_source", '{"source": {"source_kind": "PRIVATE_VALUE"}}'),
        ("inspect_source", '["PRIVATE_VALUE"]'),
        ("inspect_source", '{"source": {}, "data_root": 7}'),
    ],
)
def test_invalid_requests_exit_2_without_echoing_them(
    space: Workspace,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    content: str,
) -> None:
    """Broken JSON and invalid fields fail safely without printing the input."""
    request = write_request(tmp_path, "bad.json", content)
    assert main([command, str(request)]) == 2
    captured = capsys.readouterr()
    assert "PRIVATE_VALUE" not in captured.err and captured.out == ""
    assert main([command, str(tmp_path / "missing.json")]) == 2


def test_sigterm_during_a_slow_download_exits_143_and_keeps_the_partial(
    space: Workspace, tmp_path: Path
) -> None:
    """A real process ends with 143, prints the stopped summary and keeps bytes."""
    archive = div2k_archive()
    document = fetch_document(space, archive_spec(), run_label="cli_signal")
    request = write_request(tmp_path, "fetch.json", document.model_dump_json())
    driver = write_request(tmp_path, "driver.py", DRIVER)
    root = Path(backend_service.__file__).resolve().parents[1]
    environment = {
        name: value for name, value in os.environ.items() if "HF_" not in name
    }
    environment["PYTHONPATH"] = str(root)
    environment["STEGOLAB_LOG_DIRECTORY"] = str(space.logs)
    with LoopbackServer(archive_routes(archive, delay=30.0)) as server:
        process = subprocess.Popen(
            [sys.executable, str(driver), str(request), str(server.port)],
            cwd=tmp_path,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 60
            while process.poll() is None and time.monotonic() < deadline:
                if any(
                    path.stat().st_size > 0 for path in partial_files(space.cache_root)
                ):
                    break
                time.sleep(0.05)
            process.send_signal(signal.SIGTERM)
            output, errors = process.communicate(timeout=60)
        finally:
            if process.poll() is None:
                process.kill()
    assert process.returncode == 143, errors.decode()
    summary = DatasetFetchSummary.model_validate_json(output)
    assert (summary.status, summary.stop_signal) == ("stopped", int(signal.SIGTERM))
    assert 0 < summary.bytes_received < len(archive)
    parts = partial_files(space.cache_root)
    assert len(parts) == 1 and parts[0].stat().st_size == summary.bytes_received
    assert not (space.data_root / "div2k_sample").exists()
    assert staging_is_empty(space.data_root) and not space.output_root.exists()
    run = only_run(space, "cli_signal")
    record = json.loads((run / "dataset_fetch_run.json").read_text())
    assert record["event"] == "dataset_fetch_stopped"
    progress = json.loads((run / "fetch_progress.json").read_text())
    assert progress["phase"] == "downloading" and progress["partial_directories"]
    assert '"event": "download_partial_kept"' in (run / "events.jsonl").read_text()
