"""Uploaded archives resolve into one local asset with safe names and labels."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from backend_service.dataset_sources.plans import (
    ResolvedSource,
    materialization_identity,
    plan_for,
    suggested_source_name,
)
from backend_service.dataset_sources.transport import SecureTransport
from backend_service.dataset_sources.upload_sessions import DatasetUploadStore
from backend_service.dataset_sources.upload_source import (
    RENAMED_WARNING,
    resolve,
    safe_basename,
)
from backend_service.failures import ApplicationFailure
from schemas.dataset_common import DatasetSplit, relative_path
from schemas.dataset_sources import HttpsArchiveSourceSpec, UploadSourceSpec
from schemas.dataset_uploads import DatasetUploadCreateRequest

ARCHIVE = b"PK\x03\x04" + bytes(range(26))
ARCHIVE_SHA256 = hashlib.sha256(ARCHIVE).hexdigest()
FILE_NAME = "Photos Set.zip"


@pytest.fixture
def store(tmp_path: Path) -> DatasetUploadStore:
    """Provide an upload store under the test's own state directory."""
    return DatasetUploadStore(tmp_path / "state")


def refuse_transport(authorization: dict[str, str] | None) -> SecureTransport:
    """Fail the test if the adapter ever asks for a network transport."""
    raise AssertionError("Uploaded archives must never open a transport.")


def open_session(
    store: DatasetUploadStore, file_name: str = FILE_NAME, identifier: str = "req-1"
) -> str:
    """Create a one-part session for the sample archive; return its identifier."""
    request = DatasetUploadCreateRequest(
        client_request_identifier=identifier,
        file_name=file_name,
        total_bytes=len(ARCHIVE),
    )
    return store.create(request).upload_identifier


def completed_session(
    store: DatasetUploadStore, file_name: str = FILE_NAME, identifier: str = "req-1"
) -> str:
    """Upload and complete the sample archive; return the session identifier."""
    identifier = open_session(store, file_name, identifier)
    store.receive_chunk(identifier, 0, ARCHIVE)
    store.complete(identifier)
    return identifier


def resolve_upload(
    store: DatasetUploadStore | None, identifier: str, **overrides: Any
) -> ResolvedSource:
    """Resolve an upload spec with optional extra fields."""
    spec = UploadSourceSpec(upload_identifier=identifier, **overrides)
    return resolve(spec, transport_factory=refuse_transport, uploads=store)


def expect_failure(
    call: Callable[[], object], code: str, status_code: int
) -> ApplicationFailure:
    """Run a call that must fail with the given code and status; return it."""
    with pytest.raises(ApplicationFailure) as raised:
        call()
    assert (raised.value.code, raised.value.status_code) == (code, status_code)
    return raised.value


def test_incomplete_sessions_are_refused(store: DatasetUploadStore) -> None:
    """Missing parts report their count; unassembled uploads ask for completion."""
    identifier = open_session(store)
    failure = expect_failure(
        lambda: resolve_upload(store, identifier), "upload_incomplete", 409
    )
    assert failure.message == "1 of 1 parts are still missing."
    store.receive_chunk(identifier, 0, ARCHIVE)
    failure = expect_failure(
        lambda: resolve_upload(store, identifier), "upload_incomplete", 409
    )
    assert "Complete the upload" in failure.message
    assert not (store.root / identifier / "archive.bin").exists()


def test_complete_session_becomes_one_local_asset(store: DatasetUploadStore) -> None:
    """The archive is described by its digest and size without any download."""
    identifier = completed_session(store)
    source = resolve_upload(store, identifier, terms_reference="internal_terms")
    assert (source.source_kind, source.reference) == ("upload", "upload:Photos Set.zip")
    assert (source.requested_revision, source.resolved_revision) == (
        None,
        ARCHIVE_SHA256,
    )
    assert (source.access, source.access_guidance) == ("available", None)
    assert (source.content, source.supports_pause) == ("images", False)
    assert (source.declared_splits, source.split_mapping) == (False, {})
    assert (source.member_split_labels, source.terms_reference) == (
        {},
        "internal_terms",
    )
    assert source.warnings == (RENAMED_WARNING,)
    (asset,) = source.assets
    assert (asset.url, asset.path, asset.basename) == (
        None,
        "Photos_Set.zip",
        "Photos_Set.zip",
    )
    assert asset.local_file == store.root / identifier / "archive.bin"
    assert asset.local_file is not None and asset.local_file.read_bytes() == ARCHIVE
    assert (asset.expected_sha256, asset.expected_size) == (
        ARCHIVE_SHA256,
        len(ARCHIVE),
    )
    assert (asset.etag, asset.resumable, asset.authorization_host) == (
        None,
        False,
        None,
    )
    assert suggested_source_name(source) == "photos_set"
    plain = resolve_upload(store, completed_session(store, "sample.zip", "req-2"))
    assert plain.warnings == () and plain.assets[0].path == "sample.zip"
    assert plain.terms_reference == "not_reviewed"


def test_file_names_become_safe_basenames() -> None:
    """Spaces, accents, other scripts, leading dots and paths are neutralised."""
    assert safe_basename("Photos Set.zip") == "Photos_Set.zip"
    assert safe_basename("Träin Set.tar.gz") == "Train_Set.tar.gz"
    assert safe_basename("фото.zip") == "_.zip"
    assert safe_basename("..hidden.zip") == "hidden.zip"
    assert safe_basename(" .  .zip ") == "_.zip"
    assert safe_basename("C:\\Users\\me\\data.zip") == "data.zip"
    assert safe_basename("nested/dir/data.zip") == "data.zip"
    assert safe_basename("name:with\x00control.zip") == "name_with_control.zip"
    assert safe_basename("...") == "archive"
    assert safe_basename("") == "archive"
    assert len(safe_basename("x" * 300 + ".zip")) == 255
    for name in ("Photos Set.zip", "..hidden.zip", "фото.zip", "../../x", "..", ""):
        result = safe_basename(name)
        assert relative_path(result) == result and not result.startswith(".")


def test_archive_splits_become_labels(store: DatasetUploadStore) -> None:
    """Request folders map onto safe labels and the plan keeps them unchanged."""
    identifier = completed_session(store)
    splits: dict[str, DatasetSplit] = {"DIV2K_valid_HR": "held_out", "train": "train"}
    source = resolve_upload(store, identifier, archive_splits=splits)
    assert source.declared_splits is True
    assert source.split_mapping == {"div2k_valid_hr": "held_out", "train": "train"}
    assert source.member_split_labels == {
        "DIV2K_valid_HR": "div2k_valid_hr",
        "train": "train",
    }
    spec = UploadSourceSpec(upload_identifier=identifier, archive_splits=splits)
    plan = plan_for(
        spec, source, source_name="photos", maximum_images=10, training_intended=True
    )
    assert plan.resolved == source
    assert plan.identity == materialization_identity(spec, source, maximum_images=10)
    bare = UploadSourceSpec(upload_identifier=identifier)
    assert materialization_identity(bare, source) != plan.identity
    colliding = {"Val": "tuning", "val": "held_out"}
    expect_failure(
        lambda: resolve_upload(store, identifier, archive_splits=colliding),
        "source_split_labels",
        422,
    )


def test_missing_store_unknown_session_and_wrong_kind_are_refused(
    store: DatasetUploadStore,
) -> None:
    """Every refusal is a plain failure raised before any file is touched."""
    identifier = completed_session(store)
    expect_failure(
        lambda: resolve_upload(None, identifier), "upload_store_unavailable", 503
    )
    unknown = "upload_" + "0" * 32
    expect_failure(lambda: resolve_upload(store, unknown), "upload_not_found", 404)
    archive = HttpsArchiveSourceSpec(
        url="https://example.test/data.zip", terms_reference="https://example.test"
    )
    expect_failure(
        lambda: resolve(archive, transport_factory=refuse_transport, uploads=store),
        "upload_source_kind",
        422,
    )
