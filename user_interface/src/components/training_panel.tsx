import { useState } from 'react';
import type { useWorkspace } from '../hooks/use_workspace';
import { is_active_job } from '../hooks/use_workspace';
import type { WorkspaceCheckpoint } from '../contracts/workspace';
import type { JobSnapshot } from '../contracts/workflows';
import { TrainingSetup } from './training_setup';
import { DatasetPreparation } from './dataset_preparation';
import { JobMonitor } from './job_monitor';
import { QualityReview } from './quality_review';
import { SavedAssets } from './saved_assets';
import { RunHistory } from './run_history';
import { ErrorNotice } from './error_notice';

/** Train-first home combining durable work with the existing artifact catalog. */
export function TrainingPanel({
  state,
}: {
  state: ReturnType<typeof useWorkspace>;
}) {
  const [setup, set_setup] = useState<{ resume?: WorkspaceCheckpoint } | null>(
    null,
  );
  const [selected_run, set_selected_run] = useState('');
  const [selected_job, set_selected_job] = useState('');
  const workspace = state.workspace.data;
  const jobs = state.jobs.data?.items ?? [];
  const active_job = jobs.find(is_active_job);
  const job =
    jobs.find((item) => item.job_identifier === selected_job) ??
    active_job ??
    jobs.at(-1);
  const review_run = selected_run
    ? workspace?.runs.find((run) => run.identifier === selected_run)
    : job?.experiment_identifier
      ? workspace?.runs.find(
          (run) => run.experiment_identifier === job.experiment_identifier,
        )
      : workspace?.runs[0];
  const experiment =
    review_run?.experiment_identifier || job?.experiment_identifier || '';
  const checkpoints =
    workspace?.checkpoints.filter(
      (item) => item.experiment_identifier === experiment,
    ) ?? [];
  /** Keep the submitted worker visible while preserving the rest of the workspace. */
  function on_started(started: JobSnapshot) {
    set_setup(null);
    set_selected_job(started.job_identifier);
    set_selected_run('');
  }
  return (
    <div className="workspace_content train_workspace">
      <div className="section_heading heading_with_action">
        <div>
          <span className="eyebrow">TRAIN · OBSERVE · IMPROVE</span>
          <h1>Build. Measure. Learn.</h1>
          <p>
            Your datasets, training runs, and saved progress. One local
            workspace.
          </p>
        </div>
        <button
          className="button primary"
          disabled={
            !workspace ||
            state.jobs.isError ||
            Boolean(setup) ||
            Boolean(active_job)
          }
          onClick={() => set_setup({})}
        >
          ＋ New training run
        </button>
      </div>
      <ErrorNotice error={state.workspace.error ?? state.jobs.error} />
      {state.workspace.isError && (
        <button
          className="button secondary"
          onClick={() => {
            void state.workspace.refetch();
            void state.jobs.refetch();
          }}
        >
          Reload workspace
        </button>
      )}
      {state.workspace.isPending && (
        <p className="loading_card" role="status">
          Reading prepared datasets and saved runs…
        </p>
      )}
      {workspace && (
        <>
          <div className="overview_grid">
            <div>
              <span className="eyebrow">PREPARED DATA</span>
              <strong>
                {workspace.datasets.length}
                <small> datasets</small>
              </strong>
            </div>
            <div>
              <span className="eyebrow">SAVED PROGRESS</span>
              <strong>
                {workspace.checkpoints.length}
                <small> checkpoints</small>
              </strong>
            </div>
            <div>
              <span className="eyebrow">SHARED CPU BUDGET</span>
              <strong>
                {workspace.budget.remaining_seconds == null
                  ? 'Unknown'
                  : Math.floor(workspace.budget.remaining_seconds / 60)}
                <small> minutes left</small>
              </strong>
            </div>
            <div>
              <span className="eyebrow">MODEL STATUS</span>
              <strong className="overview_status">Experimental</strong>
            </div>
          </div>
          {workspace.budget.blocked_reason && (
            <p className="notice">{workspace.budget.blocked_reason}</p>
          )}
          {workspace.warnings.length > 0 && (
            <details className="advanced_details">
              <summary>Workspace notices ({workspace.warnings.length})</summary>
              <ul>
                {workspace.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </details>
          )}
          {setup && (
            <TrainingSetup
              key={setup.resume?.identifier ?? 'new'}
              workspace={workspace}
              resume={setup.resume}
              on_close={() => set_setup(null)}
              on_started={on_started}
            />
          )}
          {job && (
            <>
              <div className="job_selector">
                <label htmlFor="monitor_job">Monitor job</label>
                <select
                  id="monitor_job"
                  value={job.job_identifier}
                  onChange={(event) => {
                    set_selected_job(event.target.value);
                    set_selected_run('');
                  }}
                >
                  {[...jobs].reverse().map((item) => (
                    <option
                      value={item.job_identifier}
                      key={item.job_identifier}
                    >
                      {item.experiment_identifier ||
                        item.operation ||
                        item.job_identifier}{' '}
                      · {item.status} · {item.created_at}
                    </option>
                  ))}
                </select>
              </div>
              <JobMonitor
                key={job.job_identifier}
                job={job}
                stream_connected={state.stream_connected}
                confirmed={!state.jobs.isError}
              />
            </>
          )}
          <RunHistory
            workspace={workspace}
            selected={review_run?.identifier ?? ''}
            on_select={set_selected_run}
          />
          <div className="subsection_heading">
            <div>
              <span className="eyebrow">MEASURED RESULTS</span>
              <h2>
                {experiment
                  ? `Review: ${experiment}`
                  : 'Recovery and image quality'}
              </h2>
            </div>
            <span className="help_text">
              Separate measures. No automatic approval.
            </span>
          </div>
          <QualityReview
            checkpoints={checkpoints}
            run_metrics={review_run?.metrics}
          />
          <section className="lab_card">
            <div className="card_title_row">
              <div>
                <span className="eyebrow">YOUR IMAGE COLLECTIONS</span>
                <h2>Prepared datasets</h2>
              </div>
              <span className="pill muted">LOCAL FILES</span>
            </div>
            {workspace.datasets.length ? (
              <ul className="dataset_list">
                {workspace.datasets.map((dataset) => (
                  <li key={dataset.identifier}>
                    <div>
                      <strong>{dataset.name}</strong>
                      <p>
                        {dataset.image_count} images · {dataset.training_images}{' '}
                        train / {dataset.tuning_images} tune ·{' '}
                        {(dataset.prepared_bytes / 1048576).toFixed(1)} MiB
                      </p>
                      {dataset.blockers.map((reason) => (
                        <p className="dataset_blocker" key={reason}>
                          {reason}
                        </p>
                      ))}
                    </div>
                    <span
                      className={`status_pill ${dataset.compatible ? 'completed' : ''}`}
                    >
                      {dataset.compatible
                        ? 'Eligible for CPU'
                        : 'Cannot train on CPU'}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty_text">
                Prepare a local image folder to add your first dataset.
              </p>
            )}
          </section>
          <DatasetPreparation workspace={workspace} on_started={on_started} />
          <SavedAssets
            workspace={workspace}
            on_resume={(checkpoint) => {
              set_setup({ resume: checkpoint });
              window.scrollTo({ top: 0, behavior: 'instant' });
            }}
            on_started={on_started}
          />
          <p className="configuration_note">
            CPU proof profile · fixed five-minute saves · original shared
            experiment budget. Fine-tuning, checkpoint pinning, GPU jobs, and
            remote data sources need later backend support.
          </p>
        </>
      )}
    </div>
  );
}
