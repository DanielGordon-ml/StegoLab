"""Chunked upload sessions: replay, exact parts, assembly, expiry and limits."""

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import pytest

from backend_service.dataset_sources import upload_sessions
from backend_service.dataset_sources.upload_session_files import is_archive_prefix
from backend_service.dataset_sources.upload_sessions import DatasetUploadStore
from backend_service.failures import ApplicationFailure
from schemas.dataset_uploads import (
    DATASET_UPLOAD_CHUNK_BYTES,
    DatasetUploadCreateRequest,
)

MEBIBYTE = 1024**2
TOTAL_BYTES = DATASET_UPLOAD_CHUNK_BYTES + 4 * MEBIBYTE
ARCHIVE = b"PK\x03\x04" + (bytes(range(256)) * (TOTAL_BYTES // 256))[4:]
PARTS = (ARCHIVE[:DATASET_UPLOAD_CHUNK_BYTES], ARCHIVE[DATASET_UPLOAD_CHUNK_BYTES:])
ARCHIVE_SHA256 = hashlib.sha256(ARCHIVE).hexdigest()
START = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
TEMPORARY_FUNCTIONS = (
    "mkstemp mkdtemp gettempdir NamedTemporaryFile TemporaryDirectory TemporaryFile"
).split()


class Clock:
    """A clock the tests move forward by hand."""

    def __init__(self) -> None:
        """Start at a fixed instant."""
        self.now = START

    def __call__(self) -> datetime:
        """Return the current test instant."""
        return self.now


@pytest.fixture
def clock() -> Clock:
    """Provide a controllable clock."""
    return Clock()


@pytest.fixture
def store(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> DatasetUploadStore:
    """Provide a store on the test clock that may not touch the temporary directory."""

    def refuse(*arguments: object, **keywords: object) -> None:
        raise AssertionError("Upload data must stay under the state directory.")

    for name in TEMPORARY_FUNCTIONS:
        monkeypatch.setattr(tempfile, name, refuse)
    return DatasetUploadStore(tmp_path / "state", clock)


def request(
    identifier: str = "req-1",
    *,
    total_bytes: int = TOTAL_BYTES,
    expected_sha256: str | None = None,
) -> DatasetUploadCreateRequest:
    """Build a create request for the sample archive."""
    return DatasetUploadCreateRequest(
        client_request_identifier=identifier,
        file_name="sample.zip",
        total_bytes=total_bytes,
        expected_sha256=expected_sha256,
    )


def upload_all(store: DatasetUploadStore, identifier: str) -> None:
    """Send both parts of the sample archive."""
    for index, part in enumerate(PARTS):
        store.receive_chunk(identifier, index, part)


def expect_failure(
    call: Callable[[], object], code: str, status_code: int
) -> ApplicationFailure:
    """Run a call that must fail with the given code and status; return it."""
    with pytest.raises(ApplicationFailure) as raised:
        call()
    assert (raised.value.code, raised.value.status_code) == (code, status_code)
    return raised.value


def private_files(store: DatasetUploadStore) -> set[str]:
    """Check every entry under the state directory is owner-only; name the files."""
    names = set()
    for directory, directory_names, file_names in os.walk(store.root.parent):
        for name in directory_names:
            assert stat.S_IMODE((Path(directory) / name).lstat().st_mode) == 0o700
        for name in file_names:
            path = Path(directory) / name
            assert stat.S_IMODE(path.lstat().st_mode) == 0o600
            assert path.is_relative_to(store.root)
            names.add(name)
    return names


def test_create_and_replay(store: DatasetUploadStore, clock: Clock) -> None:
    """A repeated request returns the same session; a changed one is refused."""
    session = store.create(request())
    assert session.upload_identifier.startswith("upload_")
    assert (session.chunk_count, session.chunk_bytes) == (2, DATASET_UPLOAD_CHUNK_BYTES)
    assert (session.received_chunks, session.complete, session.sha256) == (
        [],
        False,
        None,
    )
    assert session.created_at == START
    assert session.expires_at == START + timedelta(hours=24)
    clock.now += timedelta(hours=1)
    assert store.create(request()) == session
    changed = request(total_bytes=TOTAL_BYTES - 1)
    expect_failure(lambda: store.create(changed), "request_identifier_conflict", 409)
    assert store.read(session.upload_identifier) == session
    assert store.create(request("req-2")).upload_identifier != session.upload_identifier


def test_parts_arrive_in_any_order_with_exact_sizes(store: DatasetUploadStore) -> None:
    """The last part may arrive first; sizes and indexes must match exactly."""
    identifier = store.create(request()).upload_identifier
    for index, data in ((0, PARTS[1]), (1, PARTS[0]), (1, PARTS[1][:-1])):
        call = partial(store.receive_chunk, identifier, index, data)
        expect_failure(call, "upload_chunk_size", 422)
    for index in (2, -1):
        call = partial(store.receive_chunk, identifier, index, PARTS[1])
        expect_failure(call, "upload_chunk_index", 422)
    assert (store.read(identifier).received_chunks, store.used_bytes()) == ([], 0)
    acknowledgement = store.receive_chunk(identifier, 1, PARTS[1])
    assert (acknowledgement.index, acknowledgement.bytes) == (1, 4 * MEBIBYTE)
    assert acknowledgement.sha256 == hashlib.sha256(PARTS[1]).hexdigest()
    assert acknowledgement.received_chunks == [1]
    assert store.read(identifier).received_chunks == [1]
    failure = expect_failure(
        lambda: store.complete(identifier), "upload_incomplete", 409
    )
    assert failure.message == "1 of 2 parts are still missing."
    expect_failure(lambda: store.archive_path(identifier), "upload_not_completed", 409)
    assert store.receive_chunk(identifier, 0, PARTS[0]).received_chunks == [0, 1]
    assert store.read(identifier).received_chunks == [0, 1]
    assert store.read(identifier).complete is False


def test_duplicate_part_is_idempotent_but_different_bytes_are_refused(
    store: DatasetUploadStore,
) -> None:
    """Re-sending the same bytes is accepted; other bytes at that index are not."""
    identifier = store.create(request()).upload_identifier
    first = store.receive_chunk(identifier, 0, PARTS[0])
    assert store.receive_chunk(identifier, 0, PARTS[0]) == first
    assert store.used_bytes() == DATASET_UPLOAD_CHUNK_BYTES
    different = b"\xff" + PARTS[0][1:]
    call = partial(store.receive_chunk, identifier, 0, different)
    expect_failure(call, "upload_chunk_mismatch", 409)
    part = store.root / identifier / "chunks" / "000000.part"
    assert part.read_bytes() == PARTS[0]
    assert store.read(identifier).received_chunks == [0]


def test_complete_assembles_a_verified_archive(store: DatasetUploadStore) -> None:
    """Completion joins the parts, checks the digest and frees the parts."""
    identifier = store.create(request(expected_sha256=ARCHIVE_SHA256)).upload_identifier
    upload_all(store, identifier)
    session = store.complete(identifier)
    assert (session.complete, session.sha256) == (True, ARCHIVE_SHA256)
    assert session.received_chunks == [0, 1]
    archive = store.archive_path(identifier)
    assert archive == store.root / identifier / "archive.bin"
    assert archive.read_bytes() == ARCHIVE
    assert not (archive.parent / "chunks").exists()
    assert store.used_bytes() == TOTAL_BYTES
    assert store.complete(identifier) == session
    assert store.read(identifier) == session
    call = partial(store.receive_chunk, identifier, 0, PARTS[0])
    expect_failure(call, "upload_already_complete", 409)
    expected_names = {"archive.bin", "session.json", "req-1", ".uploads.lock"}
    assert private_files(store) == expected_names


def test_wrong_expected_checksum_discards_the_session(
    store: DatasetUploadStore,
) -> None:
    """A digest that differs from the declared one removes everything received."""
    identifier = store.create(request(expected_sha256="0" * 64)).upload_identifier
    upload_all(store, identifier)
    expect_failure(lambda: store.complete(identifier), "upload_checksum_mismatch", 422)
    expect_failure(lambda: store.read(identifier), "upload_not_found", 404)
    assert not (store.root / identifier).exists()
    assert store.used_bytes() == 0
    replacement = store.create(request(expected_sha256="0" * 64))
    assert replacement.upload_identifier != identifier


def test_non_archive_bytes_are_refused(store: DatasetUploadStore) -> None:
    """Only zip, gzip and tar bytes may complete; anything else is discarded."""
    identifier = store.create(request(total_bytes=5)).upload_identifier
    store.receive_chunk(identifier, 0, b"hello")
    expect_failure(lambda: store.complete(identifier), "upload_not_archive", 422)
    expect_failure(lambda: store.read(identifier), "upload_not_found", 404)
    assert store.used_bytes() == 0
    assert is_archive_prefix(b"\x1f\x8b\x08" + b"\x00" * 9)
    assert is_archive_prefix(b"\x00" * 257 + b"ustar" + b"\x00" * 250)
    assert not is_archive_prefix(b"PK\x05\x06" + b"\x00" * 18)


def test_expired_sessions_are_unavailable_and_swept(
    store: DatasetUploadStore, clock: Clock
) -> None:
    """After a day a session is gone on the next touch and sweep removes the rest."""
    first = store.create(request("req-1"))
    store.receive_chunk(first.upload_identifier, 0, PARTS[0])
    clock.now += timedelta(hours=23)
    second = store.create(request("req-2"))
    clock.now += timedelta(hours=2)
    read_first = partial(store.read, first.upload_identifier)
    expect_failure(read_first, "upload_expired", 404)
    assert not (store.root / first.upload_identifier).exists()
    expect_failure(read_first, "upload_not_found", 404)
    assert store.used_bytes() == 0
    assert store.read(second.upload_identifier) == second
    replaced = store.create(request("req-1"))
    assert replaced.upload_identifier != first.upload_identifier
    assert replaced.created_at == clock.now
    clock.now += timedelta(hours=23)
    assert store.sweep() == 1
    read_second = partial(store.read, second.upload_identifier)
    expect_failure(read_second, "upload_not_found", 404)
    assert store.read(replaced.upload_identifier) == replaced
    assert store.sweep() == 0
    assert not (store.root / "requests" / "req-2").exists()


def test_storage_cap_counts_every_session(
    store: DatasetUploadStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parts and archives of all sessions share one cap; discarding frees it."""
    monkeypatch.setattr(upload_sessions, "MAXIMUM_UPLOAD_STORAGE_BYTES", 36 * MEBIBYTE)
    first = store.create(request("req-1")).upload_identifier
    store.receive_chunk(first, 1, PARTS[1])
    second = store.create(request("req-2", total_bytes=MEBIBYTE * 16 + 1))
    store.receive_chunk(first, 0, PARTS[0])
    store.receive_chunk(second.upload_identifier, 0, PARTS[0])
    assert store.used_bytes() == 36 * MEBIBYTE
    last_byte = partial(store.receive_chunk, second.upload_identifier, 1, b"x")
    expect_failure(last_byte, "upload_storage_full", 507)
    third = request("req-3", total_bytes=1)
    expect_failure(lambda: store.create(third), "upload_storage_full", 507)
    expect_failure(lambda: store.complete(first), "upload_storage_full", 507)
    store.discard(second.upload_identifier)
    assert store.complete(first).complete is True
    assert store.used_bytes() == TOTAL_BYTES
    assert store.create(request("req-3", total_bytes=16 * MEBIBYTE)).chunk_count == 1


def test_disk_reserve_is_checked_for_every_write(
    store: DatasetUploadStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each stored part and the assembly ask for the free-space reserve first."""
    calls: list[tuple[Path, int]] = []

    def record(root: Path, size: int) -> None:
        calls.append((root, size))

    monkeypatch.setattr(upload_sessions, "ensure_disk_reserve", record)
    identifier = store.create(request()).upload_identifier
    upload_all(store, identifier)
    store.receive_chunk(identifier, 1, PARTS[1])
    store.complete(identifier)
    sizes = (16 * MEBIBYTE, 4 * MEBIBYTE, 16 * MEBIBYTE)
    assert calls == [(store.root, size) for size in sizes]


def test_discard_is_safe_to_repeat(store: DatasetUploadStore) -> None:
    """Discarding removes everything and never fails for unknown identifiers."""
    identifier = store.create(request()).upload_identifier
    store.receive_chunk(identifier, 0, PARTS[0])
    store.discard(identifier)
    expect_failure(lambda: store.read(identifier), "upload_not_found", 404)
    assert store.used_bytes() == 0
    store.discard(identifier)
    store.discard("upload_" + "0" * 32)
    store.discard("../escape")
    expect_failure(lambda: store.read("../escape"), "upload_not_found", 404)
    assert store.create(request()).upload_identifier != identifier
    assert private_files(store) == {"session.json", "req-1", ".uploads.lock"}
