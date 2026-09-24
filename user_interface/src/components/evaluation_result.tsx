import evaluation_schema from '../../../contracts/entities/EvaluationReport.json';
import { create_validator } from '../contracts/validation';
import type { JobSnapshot } from '../contracts/workflows';
import { format_metric } from './quality_review';

interface EvaluationResult {
  completed: boolean;
  dataset_revision: string;
  png_trials: {
    peak_signal_to_noise_ratio: number | null;
    structural_similarity: number | null;
  }[];
  expected_png_trials: number;
  exact_recovery_rate: number | null;
  median_peak_signal_to_noise_ratio: number | null;
  median_structural_similarity: number | null;
  stopped_reason: string;
}
const validate_evaluation =
  create_validator<EvaluationResult>(evaluation_schema);

/** Show the report from this specific evaluation without relabelling older metrics. */
export function EvaluationResultCard({ job }: { job: JobSnapshot }) {
  if (job.operation !== 'evaluate' || !job.result) return null;
  if (!validate_evaluation(job.result))
    return (
      <p className="notice">
        The evaluation report format is not supported. No measurements have been
        inferred.
      </p>
    );
  const report = job.result;
  const psnr_sample_count = report.png_trials.filter(
    (trial) => trial.peak_signal_to_noise_ratio !== null,
  ).length;
  const ssim_sample_count = report.png_trials.filter(
    (trial) => trial.structural_similarity !== null,
  ).length;
  const checkpoint = job.frozen_settings.checkpoint_identifier;
  return (
    <section className="evaluation_result">
      <span className="eyebrow">THIS EVALUATION · EXPERIMENTAL</span>
      <h3>Saved evaluation result</h3>
      <p className="help_text">
        {report.completed
          ? 'Evaluation completed'
          : `Partial evaluation: ${report.stopped_reason}`}{' '}
        · {report.png_trials.length} / {report.expected_png_trials} PNG trials
      </p>
      <dl className="detail_list">
        <div>
          <dt>Exact recovery</dt>
          <dd>{format_metric(report.exact_recovery_rate, '%', true)}</dd>
        </div>
        <div>
          <dt>Image quality (PSNR)</dt>
          <dd>
            {format_metric(report.median_peak_signal_to_noise_ratio, ' dB')}
            <span className="cell_note">PSNR samples: {psnr_sample_count}</span>
          </dd>
        </div>
        <div>
          <dt>Image similarity (SSIM)</dt>
          <dd>
            {format_metric(report.median_structural_similarity)}
            <span className="cell_note">SSIM samples: {ssim_sample_count}</span>
          </dd>
        </div>
        <div>
          <dt>Dataset revision</dt>
          <dd>{report.dataset_revision}</dd>
        </div>
        {typeof checkpoint === 'string' && (
          <div>
            <dt>Checkpoint reference</dt>
            <dd>{checkpoint}</dd>
          </div>
        )}
      </dl>
      <p className="help_text">
        These results belong to this submitted evaluation. They do not approve
        the model or replace older checkpoint measurements.
      </p>
    </section>
  );
}
