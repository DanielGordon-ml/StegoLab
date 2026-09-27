"""The fetch and inspect commands find uploaded archives in the state directory."""

import io
import json
import sys
from pathlib import Path

import pytest
from fetch_fixtures import Workspace, div2k_archive, fetch_document, workspace

from backend_service.command_line import main
from backend_service.dataset_sources import fetch_downloads
from backend_service.dataset_sources.transport import SecureTransport
from backend_service.dataset_sources.upload_sessions import DatasetUploadStore
from schemas.dataset_fetch import DatasetFetchSummary
from schemas.dataset_sources import DatasetInspection
from schemas.dataset_uploads import DatasetUploadCreateRequest


@pytest.fixture
def space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """Provide a workspace whose logs folder the commands write into."""
    created = workspace(tmp_path)
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(created.logs))
    return created


def refuse(authorization: dict[str, str] | None) -> SecureTransport:
    """Fail the test if a command ever asks for a network transport."""
    raise AssertionError("Uploaded archives must never open a transport.")


def uploaded(state: Path, archive: bytes) -> str:
    """Complete one upload of the archive in the store under a state directory."""
    store = DatasetUploadStore(state)
    created = DatasetUploadCreateRequest(
        client_request_identifier="cli-upload",
        file_name="div2k.zip",
        total_bytes=len(archive),
    )
    identifier = store.create(created).upload_identifier
    store.receive_chunk(identifier, 0, archive)
    store.complete(identifier)
    return identifier


def test_commands_import_an_upload_from_the_state_directory(
    space: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both commands find uploads under the configured state directory offline."""
    archive, state = div2k_archive(), tmp_path / "state"
    monkeypatch.setenv("STEGOLAB_DATA_DIRECTORY", str(state))
    monkeypatch.setattr(fetch_downloads, "default_transport_factory", refuse)
    source = {"source_kind": "upload", "upload_identifier": uploaded(state, archive)}
    document = fetch_document(
        space, source, source_name="uploaded", prepare=False, run_label="cli_upload"
    )
    request = tmp_path / "upload.json"
    request.write_text(document.model_dump_json())
    assert main(["fetch_dataset", str(request)]) == 0
    summary = DatasetFetchSummary.model_validate_json(capsys.readouterr().out)
    assert (summary.status, summary.source_kind) == ("completed", "upload")
    assert (summary.member_count, summary.bytes_received) == (4, 0)
    assert (space.data_root / "uploaded" / "DIV2K_train_HR" / "0001.png").exists()
    inspection_request = {
        "source": source,
        "source_name": "uploaded",
        "data_root": str(space.data_root),
        "cache_root": str(space.cache_root),
    }
    stream = io.TextIOWrapper(io.BytesIO(json.dumps(inspection_request).encode()))
    monkeypatch.setattr(sys, "stdin", stream)
    assert main(["inspect_source", "-"]) == 0
    inspection = DatasetInspection.model_validate_json(capsys.readouterr().out)
    assert inspection.raw_folder == "reusable"
    assert (inspection.cached_bytes, inspection.download_bytes) == (len(archive),) * 2
