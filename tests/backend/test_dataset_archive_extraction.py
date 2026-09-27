"""Archive members are classified, bounded and streamed into the stage."""

import hashlib
import os
import tarfile
from pathlib import Path

import pytest
from archive_fixtures import (
    DIV2K_FOLDER,
    TarMember,
    ZipMember,
    build_tar,
    build_zip,
    div2k_zip,
    noisy_png,
    tiny_jpeg,
    tiny_png,
    write_archive,
)

from backend_service.dataset_sources.archive_extraction import extract_archive
from backend_service.dataset_sources.archive_members import (
    ExtractionReport,
    MemberPolicy,
    classify_member,
)
from backend_service.dataset_sources.transfer import DownloadStopped
from backend_service.dataset_storage import DatasetWriter
from backend_service.failures import ApplicationFailure

PNG = tiny_png()
POLICY = MemberPolicy()


def writer_for(tmp_path: Path) -> DatasetWriter:
    """Create the data and cache roots the writer needs and return it."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "cache").mkdir(exist_ok=True)
    return DatasetWriter(tmp_path / "data", tmp_path / "cache")


def split_for(path: str) -> str | None:
    """Label DIV2K members by their folder and leave others unlabeled."""
    return "div2k_valid_hr" if path.startswith(f"{DIV2K_FOLDER}/") else None


def extract(
    tmp_path: Path, archive: Path, policy: MemberPolicy = POLICY
) -> tuple[ExtractionReport, dict[str, bytes]]:
    """Extract into a fresh stage and return the report with the staged bytes."""
    with writer_for(tmp_path) as writer:
        report = extract_archive(
            archive, writer, policy=policy, split_for_member=split_for
        )
        staged = {
            str(path.relative_to(writer.stage)): path.read_bytes()
            for path in writer.stage.rglob("*")
            if path.is_file()
        }
    return report, staged


def reasons(report: ExtractionReport) -> dict[str, str]:
    """Map every rejected path to its reason."""
    return {rejection.path: rejection.reason for rejection in report.rejected}


def test_div2k_zip_lands_members_with_exact_paths_and_digests(tmp_path: Path) -> None:
    """Kept members reach the stage under their archive paths with true digests."""
    data, digests = div2k_zip()
    archive = write_archive(tmp_path / "archives", "DIV2K_valid_HR.zip", data)
    updates: list[tuple[int, int | None]] = []
    with writer_for(tmp_path) as writer:
        report = extract_archive(
            archive,
            writer,
            policy=POLICY,
            split_for_member=split_for,
            progress=lambda done, total: updates.append((done, total)),
        )
        for member in report.members:
            staged = writer.stage / member.path
            assert hashlib.sha256(staged.read_bytes()).hexdigest() == member.sha256
            assert staged.stat().st_size == member.size
    assert [member.path for member in report.members] == sorted(digests)
    assert {member.path: member.sha256 for member in report.members} == digests
    assert {member.split_label for member in report.members} == {"div2k_valid_hr"}
    assert {member.kind for member in report.members} == {"image"}
    assert report.rejected == ()
    assert report.total_bytes == sum(member.size for member in report.members)
    assert updates[-1] == (report.total_bytes, report.total_bytes)


def test_zip_rejections_are_recorded_with_reasons(tmp_path: Path) -> None:
    """Every unsafe or unsupported zip member is skipped for a named reason."""
    inner = build_zip([ZipMember("inner/a.png", PNG)])
    members = [
        ZipMember("good/0001.png", PNG),
        ZipMember("good/photo.jpg", tiny_jpeg()),
        ZipMember("good/notes.txt", b"caption one\n"),
        ZipMember("../x.png", PNG),
        ZipMember("/abs/x.png", PNG),
        ZipMember("good/link.png", b"target.png", symlink=True),
        ZipMember("good/secret.png", PNG, encrypted=True),
        ZipMember("good/inner.zip", inner),
        ZipMember("good/fake.png", inner),
        ZipMember("good/dup.png", PNG),
        ZipMember("good/dup.png", PNG),
        ZipMember("__MACOSX/good/._0001.png", b"\x00\x05\x16\x07"),
        ZipMember("good/.DS_Store", b"\x00\x00\x00\x01Bud1"),
        ZipMember("good/wrong.jpg", PNG),
        ZipMember("good/readme.md", b"# hello"),
        ZipMember("good/binary.txt", b"ab\x00cd"),
        ZipMember("good/huge.png", noisy_png()),
        ZipMember("good/folder/", b""),
    ]
    archive = write_archive(tmp_path / "archives", "mixed.zip", build_zip(members))
    policy = MemberPolicy(maximum_image_bytes=2048)
    report, staged = extract(tmp_path, archive, policy)
    assert reasons(report) == {
        "../x.png": "unsafe_path",
        "/abs/x.png": "unsafe_path",
        "good/link.png": "special_member",
        "good/secret.png": "encrypted_member",
        "good/inner.zip": "nested_archive",
        "good/fake.png": "nested_archive",
        "good/dup.png": "duplicate_member",
        "__MACOSX/good/._0001.png": "unsupported_file_type",
        "good/.DS_Store": "unsupported_file_type",
        "good/wrong.jpg": "content_mismatch",
        "good/readme.md": "unsupported_file_type",
        "good/binary.txt": "content_mismatch",
        "good/huge.png": "member_size_limit",
    }
    kept = {member.path: member.kind for member in report.members}
    assert kept == {
        "good/0001.png": "image",
        "good/photo.jpg": "image",
        "good/notes.txt": "text",
        "good/dup.png": "image",
    }
    assert set(staged) == set(kept)
    assert staged["good/notes.txt"] == b"caption one\n"


def test_text_members_follow_the_allowed_kinds(tmp_path: Path) -> None:
    """A policy without text rejects txt members as unsupported."""
    data = build_zip([ZipMember("a.txt", b"x"), ZipMember("a.png", PNG)])
    archive = write_archive(tmp_path / "archives", "text.zip", data)
    policy = MemberPolicy(allowed_kinds=frozenset({"image"}))
    report, staged = extract(tmp_path, archive, policy)
    assert reasons(report) == {"a.txt": "unsupported_file_type"}
    assert list(staged) == ["a.png"]
    assert classify_member("a.txt", b"", POLICY).kind == "text"
    assert classify_member("a.PNG", PNG, POLICY).kind == "image"


def test_size_lie_fails_the_run(tmp_path: Path) -> None:
    """A member whose bytes differ from the declared size rejects the archive."""
    data = build_zip([ZipMember("lie.png", PNG, declared_size=len(PNG) + 7)])
    archive = write_archive(tmp_path / "archives", "lie.zip", data)
    with pytest.raises(ApplicationFailure) as failure:
        extract(tmp_path, archive)
    assert failure.value.code == "archive_member_size"
    assert not any((tmp_path / "data" / ".staging").glob("run-*"))


def test_bomb_is_rejected_by_its_declared_size(tmp_path: Path) -> None:
    """A member that declares more bytes than the limit is skipped unread."""
    data = build_zip(
        [ZipMember("bomb.png", PNG, declared_size=10**9), ZipMember("ok.png", PNG)]
    )
    archive = write_archive(tmp_path / "archives", "bomb.zip", data)
    report, staged = extract(tmp_path, archive)
    assert reasons(report) == {"bomb.png": "member_size_limit"}
    assert report.rejected[0].declared_bytes == 10**9
    assert list(staged) == ["ok.png"]


def test_limits_fail_the_run(tmp_path: Path) -> None:
    """Member count and total byte limits stop the extraction."""
    data = build_zip([ZipMember("a.png", PNG), ZipMember("b.png", PNG)])
    archive = write_archive(tmp_path / "archives", "two.zip", data)
    with pytest.raises(ApplicationFailure) as count:
        extract(tmp_path, archive, MemberPolicy(maximum_members=1))
    assert count.value.code == "archive_limits"
    with pytest.raises(ApplicationFailure) as total:
        extract(tmp_path, archive, MemberPolicy(maximum_total_bytes=len(PNG) + 1))
    assert total.value.code == "archive_limits"
    tarred = build_tar([TarMember("a.png", PNG), TarMember("b.png", PNG)])
    archive = write_archive(tmp_path / "archives", "two.tar", tarred)
    with pytest.raises(ApplicationFailure) as tar_count:
        extract(tmp_path, archive, MemberPolicy(maximum_members=1))
    assert tar_count.value.code == "archive_limits"


def test_stop_is_checked_between_members(tmp_path: Path) -> None:
    """The stop flag ends extraction after the current member is complete."""
    data, _digests = div2k_zip()
    archive = write_archive(tmp_path / "archives", "DIV2K_valid_HR.zip", data)
    answers = iter([False, True])
    with writer_for(tmp_path) as writer:
        with pytest.raises(DownloadStopped):
            extract_archive(
                archive,
                writer,
                policy=POLICY,
                split_for_member=split_for,
                stop=lambda: next(answers, True),
            )
        staged = sorted(path.name for path in writer.stage.rglob("*.png"))
    assert staged == ["0801.png"]


def test_tar_rejections_and_gzip_streams(tmp_path: Path) -> None:
    """Tar special entries are skipped and gzip streams extract the same."""
    members = [
        TarMember("set/0001.png", PNG),
        TarMember("set/notes.txt", b"hello"),
        TarMember("set/link.png", entry_type=tarfile.SYMTYPE, linkname="0001.png"),
        TarMember("set/hard.png", entry_type=tarfile.LNKTYPE, linkname="set/0001.png"),
        TarMember("set/device.png", entry_type=tarfile.CHRTYPE),
        TarMember("set/pipe.png", entry_type=tarfile.FIFOTYPE),
        TarMember("../escape.png", PNG),
        TarMember("set/dir", entry_type=tarfile.DIRTYPE),
        TarMember("set/0001.png", PNG),
    ]
    expected = {
        "set/link.png": "special_member",
        "set/hard.png": "special_member",
        "set/device.png": "special_member",
        "set/pipe.png": "special_member",
        "../escape.png": "unsafe_path",
        "set/0001.png": "duplicate_member",
    }
    for compressed, name in ((False, "set.tar"), (True, "set.tar.gz")):
        data = build_tar(members, compressed=compressed)
        archive = write_archive(tmp_path / "archives", name, data)
        report, staged = extract(tmp_path, archive)
        assert reasons(report) == expected
        assert staged == {"set/0001.png": PNG, "set/notes.txt": b"hello"}
        assert report.members[0].sha256 == hashlib.sha256(PNG).hexdigest()


def test_unreadable_archives_fail_plainly(tmp_path: Path) -> None:
    """Damaged, truncated, unknown or linked archives are refused."""
    data, _digests = div2k_zip()
    tarred = build_tar([TarMember("a.png", PNG)], compressed=True)
    candidates = {
        "random.bin": os.urandom(600),
        "truncated.zip": data[: len(data) // 2],
        "fake.zip": b"PK\x03\x04" + os.urandom(600),
        "fake.tar.gz": b"\x1f\x8b" + os.urandom(600),
        "short.tar.gz": tarred[: len(tarred) - 40],
    }
    for name, content in candidates.items():
        archive = write_archive(tmp_path / "archives", name, content)
        with pytest.raises(ApplicationFailure) as failure:
            extract(tmp_path, archive)
        assert failure.value.code == "archive_unreadable", name
    link = tmp_path / "archives" / "link.zip"
    link.symlink_to(write_archive(tmp_path / "archives", "real.zip", data))
    with pytest.raises(ApplicationFailure) as linked:
        extract(tmp_path, link)
    assert linked.value.code == "archive_unreadable"
