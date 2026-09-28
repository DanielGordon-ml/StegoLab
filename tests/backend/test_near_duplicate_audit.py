"""Audit behaviour: leak listing, limits, benchmark blocks and report checksums."""

import json
from pathlib import Path

import pytest
from near_duplicate_fixtures import (
    audit_request,
    leaky_entries,
    resized_variant,
    smooth_image,
    snapshot,
    write_identities,
    write_revision,
)

from backend_service.failures import ApplicationFailure
from backend_service.near_duplicate_audit import (
    audit_near_duplicates,
    audit_output_path,
    load_audit_report,
    report_checksum,
)
from schemas.near_duplicate_audit import BenchmarkComparison

ZERO_SPLITS = {"train": 0, "tuning": 0, "held_out": 0}
NO_CROSS = {"train_tuning": 0, "train_held_out": 0, "tuning_held_out": 0}
IDENTITIES_FIELD = "benchmark_identities_directory"
PIXELS_FIELD = "benchmark_dataset_directory"


@pytest.fixture(scope="module")
def revision(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a revision with a within-split near duplicate and a cross-split leak."""
    root = tmp_path_factory.mktemp("audit")
    return write_revision(
        root / "datasets" / "audit_fixture" / "staging", leaky_entries()
    )


@pytest.fixture(scope="module")
def identities(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Freeze two validation members and three ranked members."""
    directory = tmp_path_factory.mktemp("identities") / "release_benchmark_v1"
    write_identities(
        directory, [139, 285], [9, 25, 30], training_source_count=2, reserve_count=1
    )
    return directory


def members(block: BenchmarkComparison | None) -> tuple[int, int, int]:
    """Read the member counts of a benchmark block."""
    assert block is not None
    return block.member_count, block.compared_members, block.missing_members


def test_cross_split_leak_is_listed_and_revision_untouched(
    revision: Path, tmp_path: Path
) -> None:
    """The leak crosses train and held_out at the gate; the revision stays as is."""
    before = snapshot(revision)
    events: list[str] = []
    report = audit_near_duplicates(
        audit_request(revision, tmp_path), progress=events.append
    )
    assert snapshot(revision) == before
    assert events == ["near_duplicate_audit_started", "near_duplicate_audit_completed"]
    assert (report.hashed_images, report.limited, report.benchmark) == (6, False, None)
    assert report.hashed_by_split == {"train": 3, "tuning": 2, "held_out": 1}
    assert report.dataset_revision == revision.name
    assert report.audit_identifier.startswith(revision.name[:12] + "_")
    assert report.pilot_ready is False
    gate = report.thresholds[2]
    assert gate.threshold == 8
    assert gate.within_split == ZERO_SPLITS | {"train": 1}
    assert gate.cross_split == NO_CROSS | {"train_held_out": 1}
    assert gate.cross_split_total == 1
    assert (gate.benchmark_by_split, gate.benchmark_total) == (None, None)
    leaks = [pair for pair in report.pairs if pair.first_split != pair.second_split]
    assert [(pair.first_identity, pair.second_identity) for pair in leaks] == [
        ("fixture:test/004.png", "fixture:training/000.png")
    ]
    assert leaks[0].distance <= 8
    assert (leaks[0].first_source, leaks[0].second_source) == ("dataset", "dataset")
    assert (report.pairs_total_at_maximum_threshold, report.pairs_truncated) == (
        2,
        False,
    )
    path = audit_output_path(tmp_path, "audit_fixture", report.audit_identifier)
    assert load_audit_report(path) == report


def test_limit_marks_the_report_limited(revision: Path, tmp_path: Path) -> None:
    """A limit keeps the first representatives by source path and says so."""
    report = audit_near_duplicates(audit_request(revision, tmp_path, limit=2))
    assert (report.limited, report.hashed_images) == (True, 2)
    assert report.hashed_by_split == {"train": 1, "tuning": 0, "held_out": 1}
    assert report.thresholds[2].cross_split_total == 1
    complete = audit_near_duplicates(
        audit_request(revision, tmp_path / "all", limit=50)
    )
    assert (complete.limited, complete.hashed_images) == (False, 6)


def test_identities_without_pixels_report_pixels_unavailable(
    revision: Path, identities: Path, tmp_path: Path
) -> None:
    """Identities alone bind the audit to the benchmark without comparing pixels."""
    report = audit_near_duplicates(
        audit_request(revision, tmp_path, **{IDENTITIES_FIELD: str(identities)})
    )
    header = json.loads((identities / "identities.json").read_bytes())
    block = report.benchmark
    assert block is not None
    assert block.identities_checksum == header["identities_checksum"]
    assert (block.status, block.benchmark_revision) == ("pixels_unavailable", None)
    assert members(block) == (5, 0, 5)
    for level in report.thresholds:
        assert (level.benchmark_by_split, level.benchmark_total) == (ZERO_SPLITS, 0)
    assert all(pair.second_source == "dataset" for pair in report.pairs)


def test_benchmark_revision_is_compared_by_member_path(
    revision: Path, identities: Path, tmp_path: Path
) -> None:
    """Records whose identity path is a member are hashed and compared."""
    benchmark = write_revision(
        tmp_path / "datasets" / "coco_fixture" / "staging",
        [
            ("val2017/000000000139.jpg", resized_variant(smooth_image(2)), None),
            ("val2017/000000000285.jpg", smooth_image(9), None),
            ("train2017/000000000009.jpg", smooth_image(10), None),
            ("extra/not_a_member.png", smooth_image(11), None),
        ],
        dataset_name="coco_fixture",
        identity_prefix="coco2017",
    )
    compare = {IDENTITIES_FIELD: str(identities), PIXELS_FIELD: str(benchmark)}
    report = audit_near_duplicates(audit_request(revision, tmp_path, **compare))
    block = report.benchmark
    assert block is not None
    assert (block.status, block.benchmark_revision) == ("compared", benchmark.name)
    assert members(block) == (5, 3, 2)
    gate = report.thresholds[2]
    assert gate.benchmark_by_split == ZERO_SPLITS | {"tuning": 1}
    assert (gate.benchmark_total, gate.cross_split_total) == (1, 1)
    matches = [pair for pair in report.pairs if pair.second_source == "benchmark"]
    assert [
        (pair.first_identity, pair.first_split, pair.second_identity, pair.second_split)
        for pair in matches
    ] == [("fixture:validation/003.png", "tuning", "val2017/000000000139.jpg", None)]
    assert (matches[0].first_source, matches[0].distance <= 4) == ("dataset", True)
    assert report.pairs_total_at_maximum_threshold == 3


def test_local_benchmark_folder_matches_members_by_source_path(
    revision: Path, identities: Path, tmp_path: Path
) -> None:
    """Local records without identities match by path; no match is a refusal."""
    local = write_revision(
        tmp_path / "datasets" / "coco_local" / "staging",
        [
            ("val2017/000000000139.jpg", resized_variant(smooth_image(2)), None),
            ("train2017/000000000030.jpg", smooth_image(12), None),
        ],
        dataset_name="coco_local",
        identity_prefix=None,
    )
    compare = {IDENTITIES_FIELD: str(identities), PIXELS_FIELD: str(local)}
    report = audit_near_duplicates(audit_request(revision, tmp_path, **compare))
    assert members(report.benchmark) == (5, 2, 3)
    assert report.thresholds[2].benchmark_total == 1
    assert [pair.second_identity for pair in report.pairs if pair.second_split is None]
    unmatched = write_revision(
        tmp_path / "datasets" / "coco_none" / "staging",
        [("extra/not_a_member.png", smooth_image(11), None)],
        dataset_name="coco_none",
        identity_prefix="coco2017",
    )
    compare[PIXELS_FIELD] = str(unmatched)
    with pytest.raises(ApplicationFailure) as failure:
        audit_near_duplicates(audit_request(revision, tmp_path / "none", **compare))
    assert failure.value.code == "audit_benchmark_unmatched"
    assert not (tmp_path / "none").exists()


def test_same_revision_and_bad_inputs_are_refused(
    revision: Path, identities: Path, tmp_path: Path
) -> None:
    """Every refusal carries a fixed code and leaves no output behind."""
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for name in ("identities.json", "members.jsonl"):
        (tampered / name).write_bytes((identities / name).read_bytes() + b"\n")
    missing = str(tmp_path / "missing")
    missing_revision = str(tmp_path / "datasets" / "coco_fixture" / "missing")
    bound: dict[str, str | int] = {IDENTITIES_FIELD: str(identities)}
    cases: list[tuple[dict[str, str | int], str]] = [
        (bound | {PIXELS_FIELD: str(revision)}, "audit_same_revision"),
        ({PIXELS_FIELD: str(revision)}, "audit_benchmark_identities_required"),
        ({"dataset_directory": missing}, "audit_dataset_invalid"),
        ({IDENTITIES_FIELD: missing}, "benchmark_invalid"),
        ({IDENTITIES_FIELD: str(tampered)}, "benchmark_invalid"),
        (bound | {PIXELS_FIELD: missing_revision}, "audit_benchmark_dataset_invalid"),
    ]
    for extra, code in cases:
        with pytest.raises(ApplicationFailure) as failure:
            audit_near_duplicates(audit_request(revision, tmp_path, **extra))
        assert failure.value.code == code
        assert "/" not in failure.value.message
    assert not (tmp_path / ".audits").exists()


def test_report_checksum_detects_tampering(revision: Path, tmp_path: Path) -> None:
    """A changed byte, an empty document or a missing file all fail to load."""
    report = audit_near_duplicates(audit_request(revision, tmp_path))
    path = audit_output_path(tmp_path, "audit_fixture", report.audit_identifier)
    assert report_checksum(report) == report.report_checksum
    content = path.read_bytes()
    assert b'"limited":false' in content
    for replacement in (content.replace(b'"limited":false', b'"limited":true'), b"{}"):
        path.write_bytes(replacement)
        with pytest.raises(ApplicationFailure) as failure:
            load_audit_report(path)
        assert failure.value.code == "audit_report_invalid"
    with pytest.raises(ApplicationFailure) as absent:
        load_audit_report(tmp_path / "absent" / "near_duplicate_audit.json")
    assert absent.value.code == "audit_report_invalid"
