"""Read the near-duplicate audit and the frozen identities during pilot preflight."""

from pathlib import Path
from typing import Final, Literal

from backend_service.benchmark_validation import validate_benchmark
from backend_service.failures import ApplicationFailure
from backend_service.near_duplicate_audit import load_audit_report
from schemas.near_duplicate_audit import (
    NearDuplicateAuditReport,
    NearDuplicateThresholdCounts,
)
from schemas.pilot_preflight import (
    PilotPreflightRequest,
    PilotReadinessCheck,
    PreflightAuditSummary,
)

PREFLIGHT_NEAR_DUPLICATE_THRESHOLD: Final = 8
NEAR_DUPLICATE_CHECK = "data_near_duplicate_audit"
BENCHMARK_CHECK = "release_benchmark"
LATER_GATE_DETAIL = "Requires a later explicit readiness or research gate."
AUDIT_GUIDANCE = (
    "Run `stegolab audit_near_duplicates` on this revision and pass its report "
    "path as near_duplicate_audit_report."
)
FREEZE_GUIDANCE = "Freeze the benchmark with `stegolab freeze_benchmark` first."
REPORT_UNREADABLE = (
    "The audit report is unreadable or changed since it was written; run the "
    "audit again."
)
REPORT_LIMITED = (
    "The audit was limited to a subset of the revision; run it without a limit "
    "before the pilot."
)
OTHER_REVISION = "The audit describes another dataset revision."
OTHER_IDENTITIES = (
    "The audit is bound to other benchmark identities than the given folder; "
    "audit again with the current identities."
)
IDENTITIES_INVALID = (
    "The benchmark identities folder exists but fails validation; restore it "
    "from version control or freeze it again."
)
AUDIT_PASSED = (
    "No cross-split near duplicates at Hamming distance 8 or less; the audit is "
    "bound to this revision."
)
CheckStatus = Literal["passed_locally", "failed", "not_run"]
READ_FAILURES = (ApplicationFailure, OSError, ValueError, MemoryError)


def _check(name: str, status: CheckStatus, detail: str) -> PilotReadinessCheck:
    """Build one readiness check with a fixed plain-language detail."""
    return PilotReadinessCheck(name=name, status=status, detail=detail)


def _gate_level(report: NearDuplicateAuditReport) -> NearDuplicateThresholdCounts:
    """Pick the pair counts at the automated distance level."""
    return next(
        level
        for level in report.thresholds
        if level.threshold == PREFLIGHT_NEAR_DUPLICATE_THRESHOLD
    )


def _summary(report: NearDuplicateAuditReport) -> PreflightAuditSummary:
    """Record what preflight read so the evidence can be traced later."""
    level = _gate_level(report)
    return PreflightAuditSummary(
        report_checksum=report.report_checksum,
        audit_identifier=report.audit_identifier,
        method_version=report.method_version,
        threshold=PREFLIGHT_NEAR_DUPLICATE_THRESHOLD,
        cross_split_pairs=level.cross_split_total,
        benchmark_status=(
            "not_included" if report.benchmark is None else report.benchmark.status
        ),
        benchmark_pairs=level.benchmark_total,
        limited=report.limited,
    )


def _audit_problem(
    report: NearDuplicateAuditReport,
    *,
    dataset_revision: str | None,
    records_checksum: str | None,
    identities_checksum: str | None,
) -> str | None:
    """Name the first reason the report cannot pass the automated gate."""
    if report.limited:
        return REPORT_LIMITED
    if (report.dataset_revision, report.records_checksum) != (
        dataset_revision,
        records_checksum,
    ):
        return OTHER_REVISION
    level = _gate_level(report)
    if level.cross_split_total > 0:
        return (
            f"{level.cross_split_total} near-duplicate pairs cross splits at distance "
            f"{PREFLIGHT_NEAR_DUPLICATE_THRESHOLD} or less; exclude them in a new "
            "revision before the pilot."
        )
    if report.benchmark is None:
        return None
    if (
        identities_checksum is not None
        and report.benchmark.identities_checksum != identities_checksum
    ):
        return OTHER_IDENTITIES
    if level.benchmark_total:
        return (
            f"{level.benchmark_total} dataset images are near duplicates of benchmark "
            f"members at distance {PREFLIGHT_NEAR_DUPLICATE_THRESHOLD} or less; "
            "exclude them in a new revision before the pilot."
        )
    return None


def near_duplicate_check(
    request: PilotPreflightRequest,
    *,
    dataset_revision: str | None,
    records_checksum: str | None,
    identities_checksum: str | None,
) -> tuple[PilotReadinessCheck, PreflightAuditSummary | None]:
    """Verify the audit report against the pilot data and gate at distance 8."""
    path = request.near_duplicate_audit_report
    try:
        if path is None or not Path(path).is_file():
            return _check(NEAR_DUPLICATE_CHECK, "not_run", AUDIT_GUIDANCE), None
        report = load_audit_report(Path(path))
    except READ_FAILURES:
        return _check(NEAR_DUPLICATE_CHECK, "failed", REPORT_UNREADABLE), None
    problem = _audit_problem(
        report,
        dataset_revision=dataset_revision,
        records_checksum=records_checksum,
        identities_checksum=identities_checksum,
    )
    if problem is not None:
        return _check(NEAR_DUPLICATE_CHECK, "failed", problem), _summary(report)
    passed = _check(NEAR_DUPLICATE_CHECK, "passed_locally", AUDIT_PASSED)
    return passed, _summary(report)


def benchmark_identities_check(
    request: PilotPreflightRequest,
) -> tuple[PilotReadinessCheck, str | None]:
    """Confirm the frozen identities verify, without approving anything later."""
    directory = request.benchmark_identities_directory
    if directory is None:
        return _check(BENCHMARK_CHECK, "not_run", LATER_GATE_DETAIL), None
    try:
        if not Path(directory).is_dir():
            return _check(BENCHMARK_CHECK, "not_run", FREEZE_GUIDANCE), None
        checksum = validate_benchmark(Path(directory)).identities_checksum
    except READ_FAILURES:
        return _check(BENCHMARK_CHECK, "failed", IDENTITIES_INVALID), None
    detail = (
        f"Identities frozen (checksum {checksum[:12]}); benchmark pixels, the 1024 "
        "preparation policy and evaluation are later gates."
    )
    return _check(BENCHMARK_CHECK, "not_run", detail), checksum
