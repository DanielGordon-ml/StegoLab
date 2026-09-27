"""Hostile sources are refused by size before they can exhaust memory or disk."""

import io
import struct
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow
import pyarrow.parquet
import pytest
from archive_fixtures import (
    TarMember,
    ZipMember,
    build_tar,
    build_zip,
    tiny_png,
    write_archive,
)

from backend_service.dataset_sources import (
    archive_members,
    materialization,
    parquet_reader,
)
from backend_service.dataset_sources.archive_extraction import extract_archive
from backend_service.dataset_sources.archive_members import (
    ExtractionReport,
    MemberPolicy,
    read_zip_directory,
)
from backend_service.dataset_sources.parquet_reader import (
    iter_image_rows,
    iter_text_rows,
)
from backend_service.dataset_sources.transfer import DownloadStopped
from backend_service.dataset_storage import DatasetWriter
from backend_service.failures import ApplicationFailure

PNG = tiny_png()
POLICY = MemberPolicy()
END_RECORD = b"PK\x05\x06"


def writer_for(tmp_path: Path) -> DatasetWriter:
    """Create the data and cache roots the writer needs and return it."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "cache").mkdir(exist_ok=True)
    return DatasetWriter(tmp_path / "data", tmp_path / "cache")


def extract(
    tmp_path: Path, archive: Path, policy: MemberPolicy = POLICY
) -> tuple[ExtractionReport, list[str]]:
    """Extract into a fresh stage and return the report with the staged paths."""
    with writer_for(tmp_path) as writer:
        report = extract_archive(
            archive, writer, policy=policy, split_for_member=lambda name: None
        )
        staged = sorted(
            str(path.relative_to(writer.stage))
            for path in writer.stage.rglob("*")
            if path.is_file()
        )
    return report, staged


def never(*arguments: Any, **options: Any) -> Any:
    """Fail the test when a guarded step is reached at all."""
    raise AssertionError("This step must be refused before it runs.")


def zip64_wrapped(data: bytes) -> bytes:
    """Rewrite a small zip so its counts live in zip64 end records only."""
    end = data.rfind(END_RECORD)
    fields = struct.unpack("<4s4H2LH", data[end : end + 22])
    entries, size, offset = fields[4], fields[5], fields[6]
    record = struct.pack(
        "<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, entries, entries, size, offset
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, end, 1)
    masked = struct.pack(
        "<4s4H2LH", END_RECORD, 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0
    )
    return data[:end] + record + locator + masked


def test_parquet_row_groups_are_bounded_before_any_row_is_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row group larger than the bound fails from the metadata alone."""
    path = tmp_path / "wide.parquet"
    images = pyarrow.array([bytes(4096)] * 4, type=pyarrow.binary())
    table = pyarrow.table({"image": images, "text": ["a" * 4096] * 4})
    pyarrow.parquet.write_table(table, path, row_group_size=2, compression="none")
    with pytest.MonkeyPatch.context() as guarded:
        guarded.setattr(parquet_reader, "MAXIMUM_ROW_GROUP_BYTES", 4096)
        guarded.setattr(pyarrow.parquet.ParquetFile, "iter_batches", never)
        readers = (
            iter_image_rows(path, "image", maximum_rows=10),
            iter_text_rows(path, "text", maximum_bytes=10**6),
        )
        for reader in readers:
            with pytest.raises(ApplicationFailure) as raised:
                next(reader)
            assert raised.value.code == "source_limits"
    assert len(list(iter_image_rows(path, "image", maximum_rows=10))) == 4
    oversized = iter_image_rows(path, "image", maximum_rows=10, maximum_bytes=4095)
    with pytest.raises(ApplicationFailure) as too_large:
        next(oversized)
    assert too_large.value.code == "source_limits"


def test_tar_members_too_large_for_any_kind_fail_before_they_are_read(
    tmp_path: Path,
) -> None:
    """A streamed member that could never be kept is not skipped by reading it."""
    members = [
        TarMember("a.png", PNG),
        TarMember("big.bin", bytes(4096)),
        TarMember("b.png", PNG),
    ]
    tarred = write_archive(tmp_path / "archives", "big.tar", build_tar(members))
    small = MemberPolicy(maximum_image_bytes=2048, maximum_text_bytes=2048)
    with pytest.raises(ApplicationFailure) as raised:
        extract(tmp_path, tarred, small)
    assert raised.value.code == "archive_limits"
    declared = MemberPolicy(maximum_total_bytes=len(PNG) + 4096 + 10)
    with pytest.raises(ApplicationFailure) as total:
        extract(tmp_path, tarred, declared)
    assert total.value.code == "archive_limits"
    report, staged = extract(tmp_path, tarred)
    assert staged == ["a.png", "b.png"] and len(report.rejected) == 1
    zipped = [ZipMember(member.name, member.data) for member in members]
    archive = write_archive(tmp_path / "archives", "big.zip", build_zip(zipped))
    report, staged = extract(tmp_path, archive, small)
    assert staged == ["a.png", "b.png"]
    assert [rejection.reason for rejection in report.rejected] == [
        "unsupported_file_type"
    ]


def test_zip_directory_is_bounded_from_the_end_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entry count and directory size are checked before the directory is parsed."""
    data = build_zip([ZipMember("a.png", PNG), ZipMember("b.png", PNG)])
    archive = write_archive(tmp_path / "archives", "two.zip", data)
    with archive.open("rb") as stream:
        directory = read_zip_directory(stream)
    assert directory.entries == 2 and 0 < directory.size_bytes < len(data)
    with pytest.MonkeyPatch.context() as guarded:
        guarded.setattr(zipfile, "ZipFile", never)
        with pytest.raises(ApplicationFailure) as count:
            extract(tmp_path, archive, MemberPolicy(maximum_members=1))
        assert count.value.code == "archive_limits"
        guarded.setattr(archive_members, "MAXIMUM_DIRECTORY_BYTES", 10)
        with pytest.raises(ApplicationFailure) as size:
            extract(tmp_path, archive)
        assert size.value.code == "archive_limits"
    wrapped = zip64_wrapped(data)
    assert zipfile.ZipFile(io.BytesIO(wrapped)).namelist() == ["a.png", "b.png"]
    with io.BytesIO(wrapped) as stream:
        assert read_zip_directory(stream) == directory
    zip64 = write_archive(tmp_path / "archives", "two64.zip", wrapped)
    report, staged = extract(tmp_path, zip64)
    assert staged == ["a.png", "b.png"] and report.rejected == ()
    with pytest.raises(ApplicationFailure) as damaged:
        with io.BytesIO(wrapped[:-10]) as stream:
            read_zip_directory(stream)
    assert damaged.value.code == "archive_unreadable"


def test_extraction_is_bounded_relative_to_the_archive_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archive that would grow far beyond its size is refused as a bomb."""
    text = ZipMember("notes.txt", b"a" * (2 * 1024**2), compress=zipfile.ZIP_DEFLATED)
    data = build_zip([ZipMember("a.png", PNG), text])
    assert len(data) < 16 * 1024
    monkeypatch.setattr(materialization, "MINIMUM_EXTRACTION_BYTES", 1024)
    bound = materialization.extraction_bound(POLICY, len(data))
    assert bound == 8 * len(data)
    assert materialization.extraction_bound(POLICY, 1) == 1024
    capped = MemberPolicy(maximum_total_bytes=7)
    assert materialization.extraction_bound(capped, 10**9) == 7
    archive = write_archive(tmp_path / "archives", "bomb.zip", data)
    with pytest.raises(ApplicationFailure) as raised:
        extract(tmp_path, archive, replace(POLICY, maximum_total_bytes=bound))
    assert raised.value.code == "archive_limits"
    assert not any((tmp_path / "data" / ".staging").glob("run-*"))
    report, staged = extract(tmp_path, archive)
    assert staged == ["a.png", "notes.txt"] and report.total_bytes > 2 * 1024**2


def test_stop_is_honoured_inside_a_large_member(tmp_path: Path) -> None:
    """The stop flag ends extraction between the chunks of one big member."""
    data = build_zip([ZipMember("big.txt", b"a" * (3 * 1024**2))])
    archive = write_archive(tmp_path / "archives", "big.zip", data)
    answers = iter([False, False, True])
    with writer_for(tmp_path) as writer:
        with pytest.raises(DownloadStopped):
            extract_archive(
                archive,
                writer,
                policy=MemberPolicy(),
                split_for_member=lambda name: None,
                stop=lambda: next(answers, True),
            )
        assert (writer.stage / "big.txt").stat().st_size < 3 * 1024**2
    assert not any((tmp_path / "data" / ".staging").glob("run-*"))
