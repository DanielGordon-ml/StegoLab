import { useState } from 'react';
import type { Workspace, WorkspaceCheckpoint } from '../contracts/workspace';
import type { JobSnapshot } from '../contracts/workflows';
import { useWorkflow } from '../hooks/use_workflow';
import { ErrorNotice } from './error_notice';
import { format_metric } from './quality_review';

/** Review registered checkpoints and export independent experimental packages. */
export function SavedAssets({
  workspace,
  on_resume,
  on_started,
}: {
  workspace: Workspace;
  on_resume: (checkpoint: WorkspaceCheckpoint) => void;
  on_started: (job: JobSnapshot) => void;
}) {
  const [checkpoint_identifier, set_checkpoint] = useState('');
  const [dataset_identifier, set_dataset] = useState('');
  const [export_name, set_export_name] = useState('001V');
  const checkpoint = workspace.checkpoints.find(
    (item) => item.identifier === checkpoint_identifier,
  );
  const workflow = useWorkflow(on_started);
  /** Bind actions to the selected checkpoint and its original experiment. */
  function submit(operation: 'evaluate' | 'export') {
    if (!checkpoint) return;
    workflow.submit({
      client_request_identifier: crypto.randomUUID(),
      operation,
      checkpoint_identifier,
      experiment_identifier: checkpoint.experiment_identifier,
      ...(operation === 'evaluate' ? { dataset_identifier } : { export_name }),
    });
  }
  return (
    <div className="quality_grid asset_grid">
      <section className="lab_card">
        <div className="card_title_row">
          <div>
            <span className="eyebrow">SAVED PROGRESS</span>
            <h2>Checkpoints</h2>
          </div>
          <span className="count_badge">{workspace.checkpoints.length}</span>
        </div>
        {workspace.checkpoints.length ? (
          <div className="form_section">
            <label htmlFor="saved_checkpoint">Saved checkpoint</label>
            <select
              id="saved_checkpoint"
              value={checkpoint_identifier}
              disabled={workflow.locked}
              onChange={(event) => {
                set_checkpoint(event.target.value);
                const selected = workspace.checkpoints.find(
                  (item) => item.identifier === event.target.value,
                );
                set_dataset(
                  workspace.datasets.find(
                    (item) => item.revision === selected?.dataset_revision,
                  )?.identifier ?? '',
                );
              }}
            >
              <option value="">Choose a checkpoint</option>
              {workspace.checkpoints.map((item) => (
                <option key={item.identifier} value={item.identifier}>
                  {item.experiment_identifier} · step {item.global_step}
                  {item.latest ? ' · latest' : ''}
                </option>
              ))}
            </select>
            {checkpoint && (
              <>
                <p className="help_text">
                  Exact recovery:{' '}
                  {format_metric(
                    checkpoint.metrics.exact_recovery_rate,
                    '%',
                    true,
                  )}{' '}
                  · Trials: {checkpoint.metrics.trial_count ?? 'Not recorded'}
                </p>
                {checkpoint.resume_blockers.length > 0 && (
                  <ul className="blocker_list">
                    {checkpoint.resume_blockers.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                )}
                <button
                  className="button secondary"
                  disabled={
                    checkpoint.resume_blockers.length > 0 || workflow.locked
                  }
                  onClick={() => on_resume(checkpoint)}
                >
                  Resume as a new job
                </button>
                <details className="advanced_details">
                  <summary>Evaluate or export</summary>
                  <label htmlFor="evaluation_dataset">Evaluation dataset</label>
                  <select
                    id="evaluation_dataset"
                    value={dataset_identifier}
                    disabled={workflow.locked}
                    onChange={(event) => set_dataset(event.target.value)}
                  >
                    <option value="">Choose the matching dataset</option>
                    {workspace.datasets.map((item) => (
                      <option key={item.identifier} value={item.identifier}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                  <button
                    className="button secondary"
                    disabled={!dataset_identifier || workflow.locked}
                    onClick={() => submit('evaluate')}
                  >
                    Evaluate checkpoint
                  </button>
                  <label htmlFor="export_name">Export version</label>
                  <input
                    id="export_name"
                    maxLength={64}
                    value={export_name}
                    disabled={workflow.locked}
                    onChange={(event) => set_export_name(event.target.value)}
                  />
                  <button
                    className="button secondary"
                    disabled={
                      workflow.locked ||
                      !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(export_name)
                    }
                    onClick={() => submit('export')}
                  >
                    Export encoder and decoder
                  </button>
                  <p className="help_text">
                    Exports remain experimental. Verification results do not
                    automatically approve them for the application.
                  </p>
                </details>
              </>
            )}
            <ErrorNotice error={workflow.error} />
            {workflow.isPending && <p role="status">Starting job…</p>}
            {workflow.uncertain && (
              <button className="button secondary" onClick={workflow.retry}>
                Retry checkpoint operation
              </button>
            )}
          </div>
        ) : (
          <p className="empty_text">
            Your first saved checkpoint will appear here.
          </p>
        )}
      </section>
      <section className="lab_card">
        <div className="card_title_row">
          <div>
            <span className="eyebrow">PORTABLE PACKAGES</span>
            <h2>Model exports</h2>
          </div>
          <span className="pill muted">EXPERIMENTAL</span>
        </div>
        {workspace.exports.length ? (
          <ul className="asset_list">
            {workspace.exports.map((item) => (
              <li key={item.identifier}>
                <div>
                  <strong>{item.name}</strong>
                  <span>{item.kind} package · experimental</span>
                </div>
                <a
                  className="button secondary"
                  href={`/api/v1/artifacts/${encodeURIComponent(item.artifact_identifier)}`}
                  download
                >
                  Download {item.kind}
                </a>
              </li>
            ))}
          </ul>
        ) : (
          <p className="empty_text">
            Export a checkpoint to create separate encoder and decoder packages.
          </p>
        )}
        <div className="preview_unavailable">
          <span aria-hidden="true">▧</span>
          <div>
            <strong>Validation previews</strong>
            <p>
              Not available for these saved artifacts. A preview needs a
              recorded checkpoint and step.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
