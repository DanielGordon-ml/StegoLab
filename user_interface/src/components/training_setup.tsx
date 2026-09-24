import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import type { Workspace, WorkspaceCheckpoint } from '../contracts/workspace';
import type { JobSnapshot, WorkflowRequest } from '../contracts/workflows';
import { check_training } from '../contracts/workflow_service';
import { useWorkflow } from '../hooks/use_workflow';
import { FixedProfile } from './fixed_profile';
import { DatasetChoice } from './training_dataset';
import { ErrorNotice } from './error_notice';

/** Guide a new or resumed run through immutable review and compatibility checks. */
export function TrainingSetup({
  workspace,
  resume,
  on_close,
  on_started,
}: {
  workspace: Workspace;
  resume?: WorkspaceCheckpoint;
  on_close: () => void;
  on_started: (job: JobSnapshot) => void;
}) {
  const [step, set_step] = useState(0);
  const [dataset_identifier, set_dataset] = useState(
    resume
      ? (workspace.datasets.find(
          (item) => item.revision === resume.dataset_revision,
        )?.identifier ?? '')
      : '',
  );
  const [name, set_name] = useState(resume?.experiment_identifier ?? '');
  const [threads, set_threads] = useState(String(resume?.cpu_threads ?? 4));
  const [stop_step, set_stop_step] = useState('1000');
  const [review_request, set_review_request] = useState<WorkflowRequest | null>(
    null,
  );
  const preflight = useMutation({ mutationFn: check_training });
  const workflow = useWorkflow(on_started);
  const dataset = workspace.datasets.find(
    (item) => item.identifier === dataset_identifier,
  );
  const valid_name = /^[a-z][a-z0-9_]{0,63}$/.test(name);
  const valid_settings =
    valid_name &&
    /^[0-9]+$/.test(threads) &&
    Number(threads) >= 1 &&
    Number(threads) <= 16 &&
    /^[0-9]+$/.test(stop_step) &&
    Number(stop_step) > (resume?.global_step ?? 0) &&
    Number(stop_step) <= 1000;
  /** Freeze values for both the preflight and subsequent start request. */
  function review() {
    const request: WorkflowRequest = {
      client_request_identifier: crypto.randomUUID(),
      operation: 'train',
      experiment_identifier: name,
      dataset_identifier,
      cpu_threads: Number(threads),
      stop_after_step: Number(stop_step),
      ...(resume ? { checkpoint_identifier: resume.identifier } : {}),
    };
    set_review_request(request);
    set_step(2);
    preflight.mutate(request);
  }
  return (
    <section className="lab_card setup_card" aria-labelledby="setup_heading">
      <div className="card_title_row">
        <div>
          <span className="eyebrow">NEW TRAINING JOB</span>
          <h2 id="setup_heading">
            {resume ? 'Resume saved progress' : 'Set up your next run'}
          </h2>
        </div>
        <button
          type="button"
          className="button secondary"
          disabled={workflow.locked}
          onClick={on_close}
        >
          Close setup
        </button>
      </div>
      <ol className="setup_steps" aria-label="Training setup progress">
        {['Choose data', 'Set up', 'Review'].map((label, index) => (
          <li key={label} aria-current={step === index ? 'step' : undefined}>
            <span>{index + 1}</span>
            {label}
          </li>
        ))}
      </ol>
      {step === 0 && (
        <DatasetChoice
          workspace={workspace}
          selected={dataset_identifier}
          locked={Boolean(resume)}
          on_select={set_dataset}
          on_continue={() => set_step(1)}
        />
      )}
      {step === 1 && (
        <form
          className="form_section"
          onSubmit={(event) => {
            event.preventDefault();
            if (valid_settings) review();
          }}
        >
          <label htmlFor="run_name">Run name</label>
          <input
            id="run_name"
            value={name}
            readOnly={Boolean(resume)}
            maxLength={64}
            onChange={(event) => set_name(event.target.value)}
            aria-describedby="run_name_help"
          />
          <p id="run_name_help" className="help_text">
            Start with a lowercase letter. Use lowercase letters, numbers, and
            underscores. Resuming keeps the original experiment budget.
          </p>
          <div className="form_columns">
            <div>
              <label htmlFor="cpu_threads">CPU threads</label>
              <input
                type="number"
                id="cpu_threads"
                readOnly={Boolean(resume)}
                min={1}
                max={16}
                step={1}
                value={threads}
                onChange={(event) => set_threads(event.target.value)}
              />
            </div>
            <div>
              <label htmlFor="stop_step">Stop at step</label>
              <input
                type="number"
                id="stop_step"
                min={(resume?.global_step ?? 0) + 1}
                max={1000}
                step={1}
                value={stop_step}
                onChange={(event) => set_stop_step(event.target.value)}
              />
            </div>
          </div>
          <FixedProfile />
          <div className="form_actions">
            <button
              type="button"
              className="button secondary"
              onClick={() => set_step(0)}
            >
              Back
            </button>
            <button className="button primary" disabled={!valid_settings}>
              Check and review →
            </button>
          </div>
        </form>
      )}
      {step === 2 && (
        <div className="form_section">
          <dl className="detail_list">
            <div>
              <dt>Run</dt>
              <dd>{name}</dd>
            </div>
            <div>
              <dt>Dataset</dt>
              <dd>{dataset?.name}</dd>
            </div>
            <div>
              <dt>Compute</dt>
              <dd>CPU · {threads} threads</dd>
            </div>
            <div>
              <dt>Stop boundary</dt>
              <dd>Step {stop_step}</dd>
            </div>
            <div>
              <dt>Starting point</dt>
              <dd>
                {resume
                  ? `Checkpoint at step ${resume.global_step}`
                  : 'New experiment'}
              </dd>
            </div>
            <div>
              <dt>Save interval</dt>
              <dd>Every 5 minutes</dd>
            </div>
          </dl>
          {preflight.isPending && (
            <p role="status">Checking data, checkpoint, runtime, and budget…</p>
          )}
          <ErrorNotice error={preflight.error} />
          {preflight.data && (
            <>
              <div
                className={`notice ${preflight.data.allowed ? 'success' : ''}`}
                role="status"
              >
                <strong>
                  {preflight.data.allowed
                    ? 'Ready to start'
                    : 'This run cannot start'}
                </strong>
                <ul>
                  {preflight.data.blockers.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
                <p>
                  {Math.floor(
                    (preflight.data.remaining_budget_seconds ?? 0) / 60,
                  )}{' '}
                  minutes available for this experiment ·{' '}
                  {preflight.data.remaining_experiment_slots ?? 'Unknown'} new
                  experiment slots
                </p>
              </div>
              {preflight.data.warnings.map((warning) => (
                <p className="help_text" key={warning}>
                  {warning}
                </p>
              ))}
            </>
          )}
          <ErrorNotice error={workflow.error} />
          {workflow.uncertain && (
            <div className="notice">
              <p>
                Start not confirmed. Retry the same request to find its result
                without creating another job.
              </p>
              <button className="button secondary" onClick={workflow.retry}>
                Retry start
              </button>
            </div>
          )}
          <div className="form_actions">
            <button
              className="button secondary"
              disabled={workflow.locked || preflight.isPending}
              onClick={() => {
                set_step(1);
                preflight.reset();
              }}
            >
              Back
            </button>
            {preflight.isError && (
              <button
                className="button secondary"
                onClick={() =>
                  review_request && preflight.mutate(review_request)
                }
              >
                Retry check
              </button>
            )}
            <button
              className="button primary"
              disabled={!preflight.data?.allowed || workflow.locked}
              onClick={() => review_request && workflow.submit(review_request)}
            >
              {workflow.isPending ? 'Starting…' : 'Start training'}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
