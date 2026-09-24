import type {
  QualityMetrics,
  WorkspaceCheckpoint,
} from '../contracts/workspace';

/** Keep missing measurements distinct from a measured zero. */
export function format_metric(
  value: number | null | undefined,
  suffix = '',
  percent = false,
) {
  return value == null
    ? 'Not measured'
    : `${(value * (percent ? 100 : 1)).toFixed(percent ? 1 : 2)}${suffix}`;
}

/** Present authenticated recovery separately from image similarity. */
export function QualityReview({
  checkpoints,
  run_metrics,
}: {
  checkpoints: WorkspaceCheckpoint[];
  run_metrics?: QualityMetrics;
}) {
  const ordered = [...checkpoints].sort(
    (first, second) => first.global_step - second.global_step,
  );
  const latest = run_metrics ?? ordered.at(-1)?.metrics;
  return (
    <>
      <p className="measurement_context">
        {run_metrics
          ? 'Cards show the selected run evaluation. Charts and table show checkpoint validation history.'
          : 'Cards show the latest checkpoint validation. No linked run evaluation is available.'}
      </p>
      <div className="quality_grid">
        <section className="lab_card metric_card">
          <span className="eyebrow">MESSAGE RECOVERY</span>
          <h3>Exact recovery</h3>
          <strong className="large_metric">
            {format_metric(latest?.exact_recovery_rate, '%', true)}
          </strong>
          <p className="help_text">
            {latest?.trial_count == null
              ? 'Samples: Not recorded'
              : `${latest.trial_count} measured trials`}
          </p>
          <MetricChart
            points={ordered}
            metric="exact_recovery_rate"
            label="Exact recovery by saved checkpoint"
          />
        </section>
        <section className="lab_card metric_card">
          <span className="eyebrow">IMAGE QUALITY</span>
          <h3>Visual similarity</h3>
          <div className="quality_numbers">
            <div>
              <strong>{format_metric(latest?.psnr, ' dB')}</strong>
              <span>PSNR · higher is better</span>
              <span>
                PSNR samples: {latest?.psnr_sample_count ?? 'Not recorded'}
              </span>
            </div>
            <div>
              <strong>{format_metric(latest?.ssim)}</strong>
              <span>SSIM · closer to 1 is better</span>
              <span>
                SSIM samples: {latest?.ssim_sample_count ?? 'Not recorded'}
              </span>
            </div>
          </div>
          <MetricChart
            points={ordered}
            metric="psnr"
            label="Image quality by saved checkpoint"
          />
        </section>
      </div>
      <details className="advanced_details">
        <summary>Checkpoint validation data table</summary>
        <div className="table_scroll">
          <table>
            <caption>
              Checkpoint validation measurements. A completed run does not
              approve a model.
            </caption>
            <thead>
              <tr>
                <th>Step</th>
                <th>Exact recovery</th>
                <th>Trials</th>
                <th>PSNR</th>
                <th>PSNR samples</th>
                <th>SSIM</th>
                <th>SSIM samples</th>
              </tr>
            </thead>
            <tbody>
              {ordered.length ? (
                ordered.map((checkpoint) => (
                  <tr key={checkpoint.identifier}>
                    <td>{checkpoint.global_step}</td>
                    <td>
                      {format_metric(
                        checkpoint.metrics.exact_recovery_rate,
                        '%',
                        true,
                      )}
                    </td>
                    <td>{checkpoint.metrics.trial_count ?? 'Not recorded'}</td>
                    <td>{format_metric(checkpoint.metrics.psnr, ' dB')}</td>
                    <td>
                      {checkpoint.metrics.psnr_sample_count ?? 'Not recorded'}
                    </td>
                    <td>{format_metric(checkpoint.metrics.ssim)}</td>
                    <td>
                      {checkpoint.metrics.ssim_sample_count ?? 'Not recorded'}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7}>No saved measurements yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

/** Draw recorded points only; blank states never imply successful measurements. */
function MetricChart({
  points,
  metric,
  label,
}: {
  points: WorkspaceCheckpoint[];
  metric: keyof QualityMetrics;
  label: string;
}) {
  const measured = points.filter((point) => point.metrics[metric] != null);
  if (!measured.length)
    return (
      <div className="chart_empty">
        <span aria-hidden="true">···</span>No linked evaluation yet
      </div>
    );
  const maximum_step = Math.max(
    1,
    ...measured.map((point) => point.global_step),
  );
  const values = measured.map((point) => point.metrics[metric]!);
  const maximum =
    metric === 'exact_recovery_rate' ? 1 : Math.max(...values) * 1.1;
  const minimum = Math.min(0, ...values);
  const positions = measured.map((point) => ({
    horizontal: 14 + (point.global_step / maximum_step) * 332,
    vertical:
      91 -
      ((point.metrics[metric]! - minimum) / Math.max(maximum - minimum, 1)) *
        76,
  }));
  return (
    <>
      <svg
        viewBox="0 0 360 112"
        className="metric_chart"
        role="img"
        aria-label={label}
      >
        <path d="M14 15H346M14 53H346M14 91H346" className="chart_grid" />
        <polyline
          points={positions
            .map((point) => `${point.horizontal},${point.vertical}`)
            .join(' ')}
          className="chart_line"
        />
        {positions.map((point, index) => (
          <circle
            key={measured[index].identifier}
            cx={point.horizontal}
            cy={point.vertical}
            r="3.5"
            className="chart_point"
          />
        ))}
        <text x="14" y="110">
          Step 0
        </text>
        <text x="346" y="110" textAnchor="end">
          Step {maximum_step}
        </text>
      </svg>
      <p className="chart_caption">Recorded evaluations only · table below</p>
    </>
  );
}
