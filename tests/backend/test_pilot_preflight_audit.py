"""Preflight reads audit reports and frozen identities as evidence, never approval."""

import json
from pathlib import Path

import pytest
from near_duplicate_fixtures import (
    resized_variant,
    smooth_image,
    write_identities,
    write_revision,
)

from backend_service.dataset_serialization import canonical_json
from backend_service.dataset_validation import load_manifest
from backend_service.near_duplicate_audit import (
    audit_near_duplicates,
    audit_output_path,
    load_audit_report,
    report_checksum,
)
from backend_service.pilot_preflight import preflight_pilot
from backend_service.pilot_preflight_audit import (
    benchmark_identities_check,
    near_duplicate_check,
)
from schemas.near_duplicate_audit import NearDuplicateAuditRequest
from schemas.pilot_preflight import (
    PilotPreflightReport,
    PilotPreflightRequest,
    PilotReadinessCheck,
    PreflightAuditSummary,
)

AUDIT = "data_near_duplicate_audit"
BENCHMARK = "release_benchmark"
LATER_GATE = "Requires a later explicit readiness or research gate."
FREEZE_FIRST = "Freeze the benchmark with `stegolab freeze_benchmark` first."
OTHER_REVISION = "The audit describes another dataset revision."
LEAK_DETAIL = (
    "1 near-duplicate pairs cross splits at distance 8 or less; exclude them in a "
    "new revision before the pilot."
)
Outcome = tuple[PilotReadinessCheck, PreflightAuditSummary | None]


@pytest.fixture(scope="module")
def revision(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a clean revision that pilot loading accepts, with a native tuning cover."""
    root = tmp_path_factory.mktemp("preflight")
    return write_revision(
        root / "datasets" / "audit_fixture" / "staging",
        [
            ("training/000.png", smooth_image(0), "training"),
            ("training/001.png", smooth_image(1), "training"),
            ("validation/002.png", smooth_image(2, 1024), "validation"),
            ("test/003.png", smooth_image(4), "test"),
        ],
    )


@pytest.fixture(scope="module")
def identities(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Freeze two validation members and three ranked members."""
    directory = tmp_path_factory.mktemp("identities") / "release_benchmark_v1"
    write_identities(
        directory, [139, 285], [9, 25, 30], training_source_count=2, reserve_count=1
    )
    return directory


@pytest.fixture(scope="module")
def report_path(
    revision: Path, identities: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Audit the clean revision once, bound to the frozen identities."""
    root = tmp_path_factory.mktemp("reports")
    return audit(revision, root, benchmark_identities_directory=str(identities))


def audit(revision: Path, root: Path, **extra: str | int) -> Path:
    """Audit a revision under the root and return the report path."""
    fields: dict[str, str | int] = {"dataset_directory": str(revision)}
    fields |= {"output_root": str(root)} | extra
    report = audit_near_duplicates(NearDuplicateAuditRequest.model_validate(fields))
    return audit_output_path(root, report.dataset_name, report.audit_identifier)


def preflight(revision: Path, **fields: str) -> PilotPreflightReport:
    """Run the real preflight against the revision with optional evidence paths."""
    request = {"dataset_directory": str(revision), "output_root": str(revision.parent)}
    return preflight_pilot(PilotPreflightRequest.model_validate(request | fields))


def statuses(report: PilotPreflightReport) -> dict[str, tuple[str, str]]:
    """Map every check name to its status and detail."""
    return {check.name: (check.status, check.detail) for check in report.checks}


def audit_check(revision: Path, path: Path, **overrides: str | None) -> Outcome:
    """Run the audit check bound to the revision unless an override says otherwise."""
    bound: dict[str, str | None] = {
        "dataset_revision": revision.name,
        "records_checksum": load_manifest(revision).records_checksum,
        "identities_checksum": None,
    }
    request = PilotPreflightRequest(
        dataset_directory=str(revision), near_duplicate_audit_report=str(path)
    )
    return near_duplicate_check(request, **(bound | overrides))


def identities_check(directory: str | None) -> tuple[PilotReadinessCheck, str | None]:
    """Run the identities check for one folder argument."""
    request = PilotPreflightRequest(
        dataset_directory="unused", benchmark_identities_directory=directory
    )
    return benchmark_identities_check(request)


def test_missing_evidence_leaves_both_checks_not_run(
    revision: Path, tmp_path: Path
) -> None:
    """Without paths, or with paths that do not exist, both checks stay not_run."""
    report = preflight(revision)
    checks = statuses(report)
    assert checks[AUDIT][0] == "not_run"
    assert "`stegolab audit_near_duplicates`" in checks[AUDIT][1]
    assert checks[BENCHMARK] == ("not_run", LATER_GATE)
    assert report.near_duplicate_audit is None
    assert report.benchmark_identities_checksum is None
    absent = preflight(
        revision,
        near_duplicate_audit_report=str(tmp_path / "absent.json"),
        benchmark_identities_directory=str(tmp_path / "absent"),
    )
    checks = statuses(absent)
    assert checks[AUDIT][0] == "not_run"
    assert checks[BENCHMARK] == ("not_run", FREEZE_FIRST)
    assert absent.pilot_ready is False and absent.gpu_checks_run is False
    assert [check.name for check in absent.checks][-2:] == [AUDIT, BENCHMARK]


def test_clean_report_passes_locally_and_binds_identities(
    revision: Path, identities: Path, report_path: Path
) -> None:
    """A clean, bound report passes; identities are recorded; nothing is approved."""
    header = json.loads((identities / "identities.json").read_bytes())
    report = preflight(
        revision,
        near_duplicate_audit_report=str(report_path),
        benchmark_identities_directory=str(identities),
    )
    checks = statuses(report)
    assert checks[AUDIT] == (
        "passed_locally",
        "No cross-split near duplicates at Hamming distance 8 or less; the audit "
        "is bound to this revision.",
    )
    assert checks[BENCHMARK] == (
        "not_run",
        f"Identities frozen (checksum {header['identities_checksum'][:12]}); "
        "benchmark pixels, the 1024 preparation policy and evaluation are later gates.",
    )
    assert report.benchmark_identities_checksum == header["identities_checksum"]
    assert report.dataset_revision == revision.name
    audit_report = load_audit_report(report_path)
    assert report.near_duplicate_audit is not None
    assert report.near_duplicate_audit.model_dump() == {
        "report_checksum": audit_report.report_checksum,
        "audit_identifier": audit_report.audit_identifier,
        "method_version": "phash_dct_32_v1",
        "threshold": 8,
        "cross_split_pairs": 0,
        "benchmark_status": "pixels_unavailable",
        "benchmark_pairs": 0,
        "limited": False,
    }
    assert report.pilot_ready is False and report.gpu_checks_run is False


def test_other_revision_limited_and_tampered_reports_fail(
    revision: Path, report_path: Path, tmp_path: Path
) -> None:
    """Reports about other data, limited runs or changed bytes never pass."""
    check, summary = audit_check(revision, report_path, dataset_revision="0" * 64)
    assert (check.status, check.detail) == ("failed", OTHER_REVISION)
    assert summary is not None and summary.cross_split_pairs == 0
    check, _ = audit_check(revision, report_path, records_checksum="0" * 64)
    assert (check.status, check.detail) == ("failed", OTHER_REVISION)
    check, summary = audit_check(revision, audit(revision, tmp_path, limit=1))
    assert check.status == "failed" and "without a limit" in check.detail
    assert summary is not None and summary.limited is True
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(
        report_path.read_bytes().replace(b'"limited":false', b'"limited":true')
    )
    check, summary = audit_check(revision, tampered)
    assert (check.status, summary) == ("failed", None)
    assert "run the audit again" in check.detail and "/" not in check.detail
    check, summary = audit_check(revision, tmp_path)
    assert (check.status, summary) == ("not_run", None)


def test_cross_split_leak_fails_the_gate(tmp_path: Path) -> None:
    """One resized copy across train and held_out is counted and named."""
    leaky = write_revision(
        tmp_path / "datasets" / "leaky" / "staging",
        [
            ("training/000.png", smooth_image(0), "training"),
            ("test/001.png", resized_variant(smooth_image(0)), "test"),
        ],
        dataset_name="leaky",
    )
    check, summary = audit_check(leaky, audit(leaky, tmp_path))
    assert (check.status, check.detail) == ("failed", LEAK_DETAIL)
    assert summary is not None
    assert (summary.cross_split_pairs, summary.benchmark_status) == (1, "not_included")
    assert summary.benchmark_pairs is None


def test_benchmark_block_must_match_identities_and_count_zero_pairs(
    revision: Path, identities: Path, report_path: Path, tmp_path: Path
) -> None:
    """Other identities or benchmark near duplicates fail the gate."""
    check, summary = audit_check(revision, report_path, identities_checksum="f" * 64)
    assert check.status == "failed" and "other benchmark identities" in check.detail
    assert summary is not None and summary.benchmark_status == "pixels_unavailable"
    audit_report = load_audit_report(report_path)
    assert audit_report.benchmark is not None
    bound = audit_report.benchmark.identities_checksum
    levels = [
        level.model_copy(update={"benchmark_total": 2})
        if level.threshold == 8
        else level
        for level in audit_report.thresholds
    ]
    forged = audit_report.model_copy(update={"thresholds": levels})
    forged.report_checksum = report_checksum(forged)
    forged_path = tmp_path / "forged.json"
    forged_path.write_bytes(canonical_json(forged.model_dump(mode="json")))
    check, summary = audit_check(revision, forged_path, identities_checksum=bound)
    assert check.status == "failed"
    assert check.detail.startswith("2 dataset images are near duplicates of benchmark")
    assert summary is not None and summary.benchmark_pairs == 2
    other = tmp_path / "other_identities"
    write_identities(other, [139], [9, 25], training_source_count=1, reserve_count=1)
    report = preflight(
        revision,
        near_duplicate_audit_report=str(report_path),
        benchmark_identities_directory=str(other),
    )
    checks = statuses(report)
    assert (checks[AUDIT][0], checks[BENCHMARK][0]) == ("failed", "not_run")
    assert report.benchmark_identities_checksum not in (None, bound)


def test_identities_check_distinguishes_missing_valid_and_invalid_folders(
    identities: Path, tmp_path: Path
) -> None:
    """A missing folder asks to freeze; a valid one names its checksum; broken fails."""
    header = json.loads((identities / "identities.json").read_bytes())
    check, checksum = identities_check(None)
    assert (check.status, check.detail, checksum) == ("not_run", LATER_GATE, None)
    check, checksum = identities_check(str(tmp_path / "absent"))
    assert (check.status, check.detail, checksum) == ("not_run", FREEZE_FIRST, None)
    check, checksum = identities_check(str(identities))
    assert (check.status, checksum) == ("not_run", header["identities_checksum"])
    assert header["identities_checksum"][:12] in check.detail
    empty = tmp_path / "empty"
    empty.mkdir()
    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ("identities.json", "members.jsonl"):
        (broken / name).write_bytes((identities / name).read_bytes() + b"\n")
    for folder in (empty, broken):
        check, checksum = identities_check(str(folder))
        assert (check.status, checksum) == ("failed", None)
        assert "/" not in check.detail
