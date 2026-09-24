"""Bounded report discovery and exact legacy evaluation associations."""

from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from backend_service.dataset_serialization import read_bounded
from backend_service.failures import ApplicationFailure
from backend_service.workspace_registry import safe_path
from schemas.evaluation import EvaluationReport
from schemas.workspace import WorkspaceMetrics


def discover(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Search only known artifact depths, limiting metadata entries per pattern."""
    result: set[Path] = set()
    for pattern in patterns:
        for path in islice(root.glob(pattern), 501):
            try:
                safe_path(path, root)
                if path.is_file():
                    result.add(path)
            except (OSError, ValueError):
                continue
    return sorted(result)


def read_metadata(path: Path) -> bytes:
    """Read a bounded regular metadata document without following links."""
    return read_bounded(path, 2 * 1024**2)


def report_metrics(report: EvaluationReport) -> WorkspaceMetrics:
    """Expose observed measurements and trial count without a release claim."""
    return WorkspaceMetrics(
        exact_recovery_rate=report.exact_recovery_rate,
        trial_count=len(report.png_trials),
        psnr=report.median_peak_signal_to_noise_ratio,
        psnr_sample_count=sum(
            trial.peak_signal_to_noise_ratio is not None for trial in report.png_trials
        ),
        ssim=report.median_structural_similarity,
        ssim_sample_count=sum(
            trial.structural_similarity is not None for trial in report.png_trials
        ),
    )


def evaluation_reports(root: Path) -> dict[tuple[str, str], EvaluationReport]:
    """Match legacy reports by exact model and dataset identity, never folder name."""
    result: dict[tuple[str, str], EvaluationReport] = {}
    paths = discover(
        root,
        (
            "logs/*/evaluation.json",
            ".runtime/*evaluation*.json",
            ".runtime/*/logs/*/evaluation.json",
        ),
    )
    for path in paths:
        try:
            report = EvaluationReport.model_validate_json(read_metadata(path))
        except (OSError, ValueError, ApplicationFailure):
            continue
        if report.kind != "proof" or not report.completed:
            continue
        key = (report.compatibility_identifier, report.dataset_revision)
        previous = result.get(key)
        if previous is None or len(report.png_trials) > len(previous.png_trials):
            result[key] = report
    return result


def report_time(path: Path) -> str:
    """Read the recorded log-folder timestamp, falling back to file modification."""
    try:
        stamp = datetime.strptime(path.parent.name[:19], "%Y-%m-%d_%H-%M-%S")
        return stamp.replace(tzinfo=UTC).isoformat()
    except ValueError:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
