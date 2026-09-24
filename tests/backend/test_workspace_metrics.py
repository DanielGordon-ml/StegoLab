"""Observed quality sample counts never substitute total attempted recoveries."""

from workspace_fixtures import evaluation_report

from backend_service.workspace_reports import report_metrics
from schemas.evaluation import PngEvaluationTrial
from schemas.workspace import WorkspaceMetrics


def test_quality_counts_follow_each_recorded_metric() -> None:
    """Missing metrics and failed trials must not inflate measured sample counts."""
    report = evaluation_report("model_identity", "dataset_revision")
    common = {
        "source_identity": "public",
        "width": 1024,
        "height": 1024,
        "fixture_kind": "maximum",
        "recovered": False,
        "elapsed_seconds": 1.0,
    }
    trials = [
        PngEvaluationTrial.model_validate(
            {**common, "peak_signal_to_noise_ratio": 28.0, "structural_similarity": 0.8}
        ),
        PngEvaluationTrial.model_validate(
            {**common, "peak_signal_to_noise_ratio": 24.0}
        ),
        PngEvaluationTrial.model_validate({**common, "error_code": "recovery_failed"}),
    ]
    metrics = report_metrics(report.model_copy(update={"png_trials": trials}))
    assert metrics.trial_count == 3
    assert metrics.psnr_sample_count == 2
    assert metrics.ssim_sample_count == 1


def test_known_empty_counts_differ_from_unknown_legacy_counts() -> None:
    """A full report records zero measured values; legacy aggregates stay unknown."""
    report = evaluation_report("model_identity", "dataset_revision")
    observed = report_metrics(report)
    assert observed.trial_count == 1
    assert observed.psnr_sample_count == observed.ssim_sample_count == 0
    legacy = WorkspaceMetrics(psnr=26.22, ssim=0.865)
    assert legacy.psnr_sample_count is None and legacy.ssim_sample_count is None
