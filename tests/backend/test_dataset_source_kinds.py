"""Contracts and foundation rules for remote and uploaded dataset sources."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend_service.application import create_application
from backend_service.dataset_scan import scan_source
from backend_service.failures import ApplicationFailure
from backend_service.hugging_face_credentials import hugging_face_token_configured
from schemas.dataset_fetch import DatasetFetchRequest
from schemas.dataset_records import DatasetPreparationRequest
from schemas.dataset_sources import (
    DatasetInspection,
    HttpsArchiveSourceSpec,
    HuggingFaceSourceSpec,
    UploadSourceSpec,
)

HUB_SOURCE = {
    "source_kind": "hugging_face",
    "repository": "Salesforce/wikitext",
    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "content": "text",
    "path_prefix": "wikitext-2-raw-v1",
    "text_column": "text",
    "terms_reference": "https://huggingface.co/datasets/Salesforce/wikitext",
}


def fetch_request(**overrides: object) -> dict[str, object]:
    """Build a valid fetch request that tests then break on purpose."""
    request: dict[str, object] = {
        "client_request_identifier": "fetch-1",
        "source": HUB_SOURCE,
        "source_name": "wikitext",
        "dataset_name": "wikitext_text",
        "prepare": False,
    }
    request.update(overrides)
    return request


def test_source_specifications_accept_pinned_sources_and_reject_unsafe_ones() -> None:
    """Every source kind validates strictly and joins one discriminated union."""
    request = DatasetFetchRequest.model_validate(fetch_request())
    assert isinstance(request.source, HuggingFaceSourceSpec)
    assert request.source.maximum_download_bytes == 25 * 1024**3
    archive = HttpsArchiveSourceSpec.model_validate(
        {
            "url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
            "archive_splits": {"DIV2K_valid_HR": "held_out"},
            "terms_reference": "https://data.vision.ee.ethz.ch/cvl/DIV2K/",
        }
    )
    assert archive.source_kind == "https_archive"
    upload = UploadSourceSpec.model_validate(
        {"upload_identifier": "upload_" + "a" * 32}
    )
    assert upload.terms_reference == "not_reviewed"
    for bad_url in (
        "http://example.org/archive.zip",
        "https://user:secret@example.org/archive.zip",
        "https://example.org/archive.zip#part",
        "https://example.org:8443/archive.zip",
        "https://example.org/archive .zip",
    ):
        with pytest.raises(ValidationError):
            HttpsArchiveSourceSpec.model_validate(
                {"url": bad_url, "terms_reference": "reviewed"}
            )
    with pytest.raises(ValidationError):
        HuggingFaceSourceSpec.model_validate({**HUB_SOURCE, "repository": "no slash"})
    with pytest.raises(ValidationError):
        HuggingFaceSourceSpec.model_validate({**HUB_SOURCE, "file_names": []})
    with pytest.raises(ValidationError):
        HttpsArchiveSourceSpec.model_validate(
            {
                "url": "https://example.org/a.zip",
                "archive_splits": {"../escape": "train"},
                "terms_reference": "reviewed",
            }
        )
    with pytest.raises(ValidationError):
        DatasetFetchRequest.model_validate(fetch_request(prepare=True))
    with pytest.raises(ValidationError):
        DatasetFetchRequest.model_validate(fetch_request(source_name="Bad Name"))
    with pytest.raises(ValidationError):
        DatasetFetchRequest.model_validate(fetch_request(source={"source_kind": "ftp"}))


def test_inspection_only_claims_disk_sufficiency_with_an_estimate() -> None:
    """An unknown download size must not come with a disk verdict."""
    document = {
        "source_kind": "https_archive",
        "reference": "https://example.org/a.zip",
        "suggested_source_name": "a",
        "access": "available",
        "content": "images",
        "asset_count": 1,
        "supports_pause": False,
        "declared_splits": False,
        "materialization_identity": "0" * 64,
        "free_disk_bytes": 1,
        "server_token_configured": False,
    }
    assert DatasetInspection.model_validate(document).disk_sufficient is None
    with pytest.raises(ValidationError):
        DatasetInspection.model_validate({**document, "disk_sufficient": True})


def test_remote_preparation_requests_need_explicit_provenance(tmp_path: Path) -> None:
    """Remote kinds never inherit the UHD-IQA reference and need no metadata."""
    base = {"source_directory": str(tmp_path), "source_kind": "https_archive"}
    with pytest.raises(ValidationError):
        DatasetPreparationRequest.model_validate(base)
    request = DatasetPreparationRequest.model_validate(
        {**base, "source_url": "https://example.org/a.zip", "terms_reference": "r"}
    )
    assert request.metadata_file is None
    assert request.training_intended is True
    local = DatasetPreparationRequest.model_validate(
        {"source_directory": str(tmp_path), "source_kind": "local", "dataset_name": "x"}
    )
    assert (local.source_url, local.terms_reference) == ("local", "not_reviewed")


def test_hidden_entries_are_sidecars_and_hidden_folders_are_skipped(
    tmp_path: Path,
) -> None:
    """A provenance marker or staging folder never counts as a rejected image."""
    source = tmp_path / "source"
    (source / ".staging").mkdir(parents=True)
    (source / ".staging" / "junk.png").write_bytes(b"x")
    (source / ".stegolab_source.json").write_text("{}")
    (source / "one.png").write_bytes(b"x")
    (source / "notes.json").write_text("{}")
    inventory = scan_source(source)
    kinds = {item.relative_path: item.kind for item in inventory.files}
    assert kinds == {
        ".stegolab_source.json": "sidecar",
        "one.png": "image",
        "notes.json": "unsupported",
    }
    assert inventory.scanned_entries == 4


def test_token_presence_is_reported_without_its_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capabilities say whether a server token exists and nothing more."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_FILE", raising=False)
    assert not hugging_face_token_configured()
    monkeypatch.setenv("HF_TOKEN", "   ")
    assert not hugging_face_token_configured()
    token_file = tmp_path / "token"
    token_file.write_text("hf_private_sentinel\n")
    monkeypatch.setenv("HF_TOKEN_FILE", str(token_file))
    assert hugging_face_token_configured()
    with TestClient(
        create_application(tmp_path / "data", tmp_path / "logs", tmp_path / "root"),
        raise_server_exceptions=False,
    ) as client:
        body = client.get("/api/v1/capabilities").json()
    assert body["hugging_face_token_configured"] is True
    assert body["dataset_source_kinds"] == ["server_folder"]
    assert body["maximum_dataset_upload_bytes"] == 2 * 1024**3
    assert "hf_private_sentinel" not in json.dumps(body)


def test_workspace_lists_visible_source_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The data folder is offered as a source and lists its real subfolders."""
    root = tmp_path / "root"
    (root / "data" / "alpha").mkdir(parents=True)
    (root / "data" / ".staging").mkdir()
    (root / "data" / "beta.png").write_bytes(b"x")
    os.symlink(root / "data" / "alpha", root / "data" / "link")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("STEGOLAB_SOURCE_ROOTS", json.dumps({"Mounted": str(other)}))
    with TestClient(
        create_application(tmp_path / "data", tmp_path / "logs", root),
        raise_server_exceptions=False,
    ) as client:
        sources = client.get("/api/v1/workspace").json()["sources"]
    labels = {item["label"]: item["folders"] for item in sources}
    assert labels == {"Mounted": [], "Downloaded and uploaded sources": ["alpha"]}


def test_failures_keep_fixed_messages() -> None:
    """The shared failure type carries plain codes for the new modules."""
    failure = ApplicationFailure("dataset_space", "Not enough free space.", 507)
    assert (failure.code, failure.status_code) == ("dataset_space", 507)
