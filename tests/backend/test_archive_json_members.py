"""Archive members named ``.json`` are text members under a sniff and the cap."""

import codecs
from hashlib import sha256
from pathlib import Path

from archive_fixtures import ZipMember, build_zip, tiny_jpeg, write_archive

from backend_service.dataset_sources.archive_extraction import extract_archive
from backend_service.dataset_sources.archive_members import (
    TEXT_EXTENSIONS,
    ExtractionReport,
    MemberPolicy,
    classify_member,
)
from backend_service.dataset_storage import DatasetWriter

POLICY = MemberPolicy()
CAPTIONS = (
    b'{"info": {"year": 2017}, '
    b'"images": [{"id": 139, "file_name": "000000000139.jpg"}]}'
)
CAPTIONS_NAME = "annotations/captions_val2017.json"
INSTANCES_NAME = "annotations/instances_train2017.json"
IDENTIFIERS_NAME = "annotations/image_identifiers.json"
IMAGE_NAME = "val2017/000000000139.jpg"
INSTANCES_TRAIN_BYTES = 470 * 1000**2


def extract(
    tmp_path: Path, archive: Path, policy: MemberPolicy = POLICY
) -> tuple[ExtractionReport, dict[str, bytes]]:
    """Extract into a fresh stage and return the report with the staged bytes."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "cache").mkdir(exist_ok=True)
    with DatasetWriter(tmp_path / "data", tmp_path / "cache") as writer:
        report = extract_archive(
            archive, writer, policy=policy, split_for_member=lambda _name: None
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


def test_json_members_are_kept_as_text_byte_for_byte(tmp_path: Path) -> None:
    """Objects, arrays and byte-order-marked json land unchanged as text."""
    members = {
        CAPTIONS_NAME: CAPTIONS,
        IDENTIFIERS_NAME: b"[139, 285, 632]",
        "annotations/marked.json": codecs.BOM_UTF8 + b" \r\n\t" + CAPTIONS,
        "annotations/UPPER.JSON": b"\n[]",
        IMAGE_NAME: tiny_jpeg(),
    }
    zipped = build_zip([ZipMember(name, data) for name, data in members.items()])
    archive = write_archive(tmp_path / "archives", "annotations.zip", zipped)
    report, staged = extract(tmp_path, archive)
    assert "json" in TEXT_EXTENSIONS
    assert report.rejected == ()
    assert staged == members
    kinds = {member.path: member.kind for member in report.members}
    assert kinds == {**dict.fromkeys(members, "text"), IMAGE_NAME: "image"}
    expected = {name: sha256(data).hexdigest() for name, data in members.items()}
    assert {member.path: member.sha256 for member in report.members} == expected
    assert report.total_bytes == sum(len(data) for data in members.values())


def test_json_that_does_not_open_an_object_or_array_is_a_content_mismatch(
    tmp_path: Path,
) -> None:
    """The first byte after the mark and whitespace decides; txt is not sniffed."""
    bad = {
        "bad/garbage.json": b'garbage {"images": []}',
        "bad/page.json": b"<!DOCTYPE html><html></html>",
        "bad/empty.json": b"",
        "bad/blank.json": b" \r\n\t",
        "bad/mark_only.json": codecs.BOM_UTF8,
        "bad/marked_garbage.json": codecs.BOM_UTF8 + b"garbage",
        "bad/zero_byte.json": b'{"images": [\x00]}',
        "bad/number.json": b"42",
    }
    good = {"good/object.json": b"{}", "good/notes.txt": b'garbage {"images": []}'}
    members = [ZipMember(name, data) for name, data in {**bad, **good}.items()]
    archive = write_archive(tmp_path / "archives", "garbage.zip", build_zip(members))
    report, staged = extract(tmp_path, archive)
    assert reasons(report) == dict.fromkeys(bad, "content_mismatch")
    assert staged == good
    image_only = MemberPolicy(allowed_kinds=frozenset({"image"}))
    rejected = classify_member("a.json", b"{}", image_only)
    assert rejected.reason == "unsupported_file_type"


def test_text_cap_applies_to_json_members(tmp_path: Path) -> None:
    """A json member past the 256 MiB text cap is skipped unread; small ones pass."""
    assert POLICY.maximum_bytes_for("text") == 256 * 1024**2
    huge = ZipMember(INSTANCES_NAME, b"{}", declared_size=INSTANCES_TRAIN_BYTES)
    zipped = build_zip([huge, ZipMember(CAPTIONS_NAME, CAPTIONS)])
    archive = write_archive(tmp_path / "archives", "trainval.zip", zipped)
    report, staged = extract(tmp_path, archive)
    assert reasons(report) == {INSTANCES_NAME: "member_size_limit"}
    assert report.rejected[0].declared_bytes == INSTANCES_TRAIN_BYTES
    assert staged == {CAPTIONS_NAME: CAPTIONS}
    identifiers = ZipMember(IDENTIFIERS_NAME, b"[139]")
    zipped = build_zip([ZipMember(CAPTIONS_NAME, CAPTIONS), identifiers])
    archive = write_archive(tmp_path / "archives", "small_cap.zip", zipped)
    small = MemberPolicy(maximum_text_bytes=len(CAPTIONS) - 1)
    report, staged = extract(tmp_path, archive, small)
    assert reasons(report) == {CAPTIONS_NAME: "member_size_limit"}
    assert staged == {IDENTIFIERS_NAME: b"[139]"}
