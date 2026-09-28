"""Where audit reports land: default root, refusals, symlinks and identifiers."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from near_duplicate_fixtures import (
    audit_request,
    leaky_entries,
    snapshot,
    write_identities,
    write_revision,
)

from backend_service import near_duplicate_audit
from backend_service.failures import ApplicationFailure
from backend_service.near_duplicate_audit import (
    audit_identifier,
    audit_near_duplicates,
    audit_output_path,
    load_audit_report,
)
from schemas.near_duplicate_audit import NearDuplicateAuditRequest

FIXED_TIME = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
VARYING = {"audit_identifier", "elapsed_seconds", "peak_process_mebibytes"}
IDENTITIES_FIELD = "benchmark_identities_directory"
PIXELS_FIELD = "benchmark_dataset_directory"


@pytest.fixture(scope="module")
def revision(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the leaky revision once for every output test."""
    root = tmp_path_factory.mktemp("output")
    return write_revision(
        root / "datasets" / "audit_fixture" / "staging", leaky_entries()
    )


@pytest.fixture(scope="module")
def identities(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Freeze a tiny identities folder for requests that name a benchmark."""
    directory = tmp_path_factory.mktemp("identities") / "release_benchmark_v1"
    write_identities(
        directory, [139], [9, 25], training_source_count=1, reserve_count=1
    )
    return directory


def test_output_root_defaults_beside_the_dataset_folder(
    revision: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an output root the report lands next to the dataset name folder."""
    monkeypatch.setattr(near_duplicate_audit, "current_time", lambda: FIXED_TIME)
    report = audit_near_duplicates(
        NearDuplicateAuditRequest(dataset_directory=str(revision))
    )
    identifier = audit_identifier(revision.name, FIXED_TIME)
    assert (
        report.audit_identifier
        == identifier
        == revision.name[:12] + "_20260928T120000Z"
    )
    expected = audit_output_path(revision.parent.parent, "audit_fixture", identifier)
    assert load_audit_report(expected) == report


def test_relative_revision_path_keeps_the_report_beside_the_dataset_folder(
    revision: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare revision name from inside the dataset folder never nests reports."""
    monkeypatch.chdir(revision.parent)
    monkeypatch.setattr(
        near_duplicate_audit, "current_time", lambda: FIXED_TIME.replace(hour=13)
    )
    report = audit_near_duplicates(
        NearDuplicateAuditRequest(dataset_directory=revision.name)
    )
    expected = audit_output_path(
        revision.parent.parent, "audit_fixture", report.audit_identifier
    )
    assert load_audit_report(expected) == report
    assert sorted(path.name for path in revision.parent.iterdir()) == [revision.name]


def test_output_root_inside_a_dataset_folder_is_refused(
    revision: Path, identities: Path, tmp_path: Path
) -> None:
    """Roots that are a revision or a dataset name folder are refused untouched."""
    before = snapshot(revision)
    benchmark_folder = tmp_path / "datasets" / "coco_fixture"
    benchmark: dict[str, str | int] = {
        IDENTITIES_FIELD: str(identities),
        PIXELS_FIELD: str(benchmark_folder / ("b" * 64)),
    }
    cases: list[tuple[Path, dict[str, str | int]]] = [
        (revision, {}),
        (revision / "images", {}),
        (revision.parent, {}),
        (revision.parent / "notes", {}),
        (benchmark_folder, benchmark),
    ]
    for root, extra in cases:
        with pytest.raises(ApplicationFailure) as failure:
            audit_near_duplicates(audit_request(revision, root, **extra))
        assert failure.value.code == "audit_output_overlaps"
        assert "/" not in failure.value.message
    assert snapshot(revision) == before
    assert list(revision.parent.rglob(".audits")) == []
    assert list(tmp_path.rglob(".audits")) == []


def test_symlinked_audits_folder_is_refused(revision: Path, tmp_path: Path) -> None:
    """A symbolic link in the report path never redirects the write."""
    (tmp_path / ".audits").symlink_to(revision / "images", target_is_directory=True)
    before = snapshot(revision)
    with pytest.raises(ApplicationFailure) as failure:
        audit_near_duplicates(audit_request(revision, tmp_path))
    assert failure.value.code == "audit_storage"
    assert snapshot(revision) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [".audits"]


def test_existing_output_folder_is_refused(
    revision: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audit never writes into a folder that already exists."""
    monkeypatch.setattr(near_duplicate_audit, "current_time", lambda: FIXED_TIME)
    path = audit_output_path(
        tmp_path, "audit_fixture", audit_identifier(revision.name, FIXED_TIME)
    )
    path.parent.mkdir(parents=True)
    with pytest.raises(ApplicationFailure) as failure:
        audit_near_duplicates(audit_request(revision, tmp_path))
    assert failure.value.code == "audit_output_exists"
    assert list(path.parent.iterdir()) == []


def test_second_run_creates_a_new_identifier_folder(
    revision: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each run gets its own folder; the evidence itself is reproducible."""
    times = iter([FIXED_TIME, FIXED_TIME.replace(second=1)])
    monkeypatch.setattr(near_duplicate_audit, "current_time", lambda: next(times))
    first = audit_near_duplicates(audit_request(revision, tmp_path))
    second = audit_near_duplicates(audit_request(revision, tmp_path))
    assert first.audit_identifier != second.audit_identifier
    folders = {path.name for path in (tmp_path / ".audits" / "audit_fixture").iterdir()}
    assert folders == {first.audit_identifier, second.audit_identifier}
    excluded = VARYING | {"report_checksum"}
    assert first.model_dump(exclude=excluded) == second.model_dump(exclude=excluded)


def test_identifier_carries_the_start_time(
    revision: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The naming moment is read before hashing starts, not after it finishes."""
    events: list[str] = []

    def timed() -> datetime:
        events.append("current_time")
        return FIXED_TIME

    monkeypatch.setattr(near_duplicate_audit, "current_time", timed)
    report = audit_near_duplicates(
        audit_request(revision, tmp_path), progress=events.append
    )
    assert events == [
        "current_time",
        "near_duplicate_audit_started",
        "near_duplicate_audit_completed",
    ]
    assert report.audit_identifier == audit_identifier(revision.name, FIXED_TIME)
