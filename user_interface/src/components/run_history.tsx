import { useState } from 'react';
import type { Workspace } from '../contracts/workspace';
import { format_metric } from './quality_review';

/** Keep recorded runs visible, including runs that predate the GUI. */
export function RunHistory({
  workspace,
  selected,
  on_select,
}: {
  workspace: Workspace;
  selected: string;
  on_select: (identifier: string) => void;
}) {
  const [comparison, set_comparison] = useState('');
  const first = workspace.runs.find((run) => run.identifier === selected);
  const second = workspace.runs.find((run) => run.identifier === comparison);
  return (
    <section className="lab_card">
      <div className="card_title_row">
        <div>
          <span className="eyebrow">EXPERIMENT LOG</span>
          <h2>Recent runs</h2>
        </div>
        <span className="count_badge">{workspace.runs.length}</span>
      </div>
      {workspace.runs.length ? (
        <>
          <div className="table_scroll">
            <table>
              <caption className="visually_hidden">
                Recorded runs and their saved measurements
              </caption>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Step</th>
                  <th>Exact recovery</th>
                  <th>Trials</th>
                  <th>Review</th>
                </tr>
              </thead>
              <tbody>
                {workspace.runs.map((run) => (
                  <tr
                    key={run.identifier}
                    aria-selected={selected === run.identifier}
                  >
                    <td>
                      <strong>{run.experiment_identifier}</strong>
                      <span className="cell_note">
                        {new Date(run.created_at).toLocaleDateString()}
                      </span>
                    </td>
                    <td>
                      <span className="status_pill">
                        {run.status.replaceAll('_', ' ')}
                      </span>
                    </td>
                    <td>{run.global_step}</td>
                    <td>
                      {format_metric(
                        run.metrics.exact_recovery_rate,
                        '%',
                        true,
                      )}
                    </td>
                    <td>{run.metrics.trial_count ?? 'Not recorded'}</td>
                    <td>
                      <button
                        className="text_button"
                        onClick={() => on_select(run.identifier)}
                        aria-label={`Review ${run.experiment_identifier}`}
                      >
                        Review →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details className="advanced_details compare_details">
            <summary>Compare saved runs</summary>
            <label htmlFor="comparison_run">
              Compare with {first?.experiment_identifier ?? 'the selected run'}
            </label>
            <select
              id="comparison_run"
              value={comparison}
              onChange={(event) => set_comparison(event.target.value)}
            >
              <option value="">Choose another run</option>
              {workspace.runs
                .filter((run) => run.identifier !== first?.identifier)
                .map((run) => (
                  <option key={run.identifier} value={run.identifier}>
                    {run.experiment_identifier}
                  </option>
                ))}
            </select>
            {first && second && (
              <>
                <p className="notice">
                  {first.dataset_revision &&
                  first.dataset_revision === second.dataset_revision
                    ? 'Dataset revisions match.'
                    : 'Unmatched dataset revisions.'}{' '}
                  Profile and payload compatibility are not recorded in this
                  catalog. This is an unmatched comparison; do not rank these
                  runs from these results alone.
                </p>
                <div className="table_scroll">
                  <table>
                    <caption>Unmatched run comparison</caption>
                    <thead>
                      <tr>
                        <th>Measure</th>
                        <th>{first.experiment_identifier}</th>
                        <th>{second.experiment_identifier}</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th>Exact recovery</th>
                        <td>
                          {format_metric(
                            first.metrics.exact_recovery_rate,
                            '%',
                            true,
                          )}
                        </td>
                        <td>
                          {format_metric(
                            second.metrics.exact_recovery_rate,
                            '%',
                            true,
                          )}
                        </td>
                      </tr>
                      <tr>
                        <th>Trials</th>
                        <td>{first.metrics.trial_count ?? 'Not recorded'}</td>
                        <td>{second.metrics.trial_count ?? 'Not recorded'}</td>
                      </tr>
                      <tr>
                        <th>PSNR</th>
                        <td>{format_metric(first.metrics.psnr, ' dB')}</td>
                        <td>{format_metric(second.metrics.psnr, ' dB')}</td>
                      </tr>
                      <tr>
                        <th>PSNR samples</th>
                        <td>
                          {first.metrics.psnr_sample_count ?? 'Not recorded'}
                        </td>
                        <td>
                          {second.metrics.psnr_sample_count ?? 'Not recorded'}
                        </td>
                      </tr>
                      <tr>
                        <th>SSIM</th>
                        <td>{format_metric(first.metrics.ssim)}</td>
                        <td>{format_metric(second.metrics.ssim)}</td>
                      </tr>
                      <tr>
                        <th>SSIM samples</th>
                        <td>
                          {first.metrics.ssim_sample_count ?? 'Not recorded'}
                        </td>
                        <td>
                          {second.metrics.ssim_sample_count ?? 'Not recorded'}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </details>
        </>
      ) : (
        <p className="empty_text">
          No runs yet. Choose a prepared dataset to start your first experiment.
        </p>
      )}
    </section>
  );
}
