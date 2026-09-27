import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { JobActionRequest, JobSnapshot } from '../contracts/workflows';
import { act_on_job } from '../contracts/workflow_service';
import { RequestFailure } from '../contracts/errors';
import { DatasetSummaryCard } from './dataset_summary_card';
import { EvaluationResultCard } from './evaluation_result';
import { ErrorNotice } from './error_notice';
import { phase_label, TransferProgress } from './job_transfer';

type ActionKind = JobActionRequest['action'];
const TRAINING_LABELS: Record<ActionKind, string> = {
  stop: 'Stop and save',
  cancel: 'Cancel job',
  pause: 'Pause and save',
  resume: 'Resume job',
};
const DOWNLOAD_LABELS: Record<ActionKind, string> = {
  stop: 'Stop download',
  cancel: 'Cancel job',
  pause: 'Pause download',
  resume: 'Resume download',
};

/** Explain a pending action in words that match the kind of job. */
function requested_action_text(job: JobSnapshot, fetch_job: boolean) {
  if (fetch_job)
    return job.requested_action === 'cancel'
      ? 'Cancelling and cleaning up. Waiting for the backend to confirm.'
      : job.requested_action === 'pause'
        ? 'Pausing the download. Waiting for the backend to confirm.'
        : 'Action requested. Waiting for the backend to confirm.';
  return job.requested_action === 'stop' || job.requested_action === 'pause'
    ? 'Saving progress before stopping. Wait for the saved result.'
    : 'Action requested. Waiting for the worker.';
}

/** Count extracted files against the total when the backend knows it. */
function file_progress_text(metrics: Record<string, number>) {
  const completed: number | undefined = metrics.files_completed;
  const total: number | undefined = metrics.files_total;
  if (total != null) return `${completed ?? 0} of ${total}`;
  if (completed) return `${completed} so far`;
  return 'Not counted yet';
}

/** Monitor durable work and request only actions the server currently allows. */
export function JobMonitor({
  job,
  stream_connected,
  confirmed = true,
}: {
  job: JobSnapshot;
  stream_connected: boolean;
  confirmed?: boolean;
}) {
  const client = useQueryClient();
  const [attempt, set_attempt] = useState<JobActionRequest | null>(null);
  const action = useMutation({
    mutationFn: (request: JobActionRequest) =>
      act_on_job(job.job_identifier, request),
    onSuccess: () => {
      set_attempt(null);
      void client.invalidateQueries({ queryKey: ['jobs'] });
      void client.invalidateQueries({ queryKey: ['workspace'] });
    },
  });
  const uncertain =
    action.isError &&
    action.error instanceof RequestFailure &&
    action.error.uncertain;
  const fetch_job = job.operation === 'fetch_dataset';
  const labels = fetch_job ? DOWNLOAD_LABELS : TRAINING_LABELS;
  const running = job.status === 'queued' || job.status === 'running';
  const pending_action = running && Boolean(job.requested_action);
  const global_step = job.metrics?.global_step;
  const connection_text = !confirmed
    ? 'Job status unavailable. Showing the last saved snapshot.'
    : stream_connected
      ? 'Live updates connected'
      : running
        ? 'Checking saved status every 5 seconds'
        : 'Saved job status';
  /** Keep action identity stable when a network failure leaves the result unknown. */
  function request_action(kind: ActionKind) {
    const request = {
      client_request_identifier: crypto.randomUUID(),
      action: kind,
    };
    set_attempt(request);
    action.mutate(request);
  }
  return (
    <section
      className="lab_card monitor_card"
      aria-labelledby="monitor_heading"
    >
      <div className="card_title_row">
        <div>
          <span className="eyebrow">JOB MONITOR</span>
          <h2 id="monitor_heading">
            {fetch_job
              ? 'Dataset download'
              : job.experiment_identifier ||
                job.operation?.replaceAll('_', ' ') ||
                'Saved job'}
          </h2>
        </div>
        <span className={`status_pill ${job.status}`}>
          {!confirmed && 'Last known: '}
          {job.status.replaceAll('_', ' ')}
        </span>
      </div>
      <p className="phase_text">
        {fetch_job ? phase_label(job.phase) : job.phase.replaceAll('_', ' ')}
      </p>
      {fetch_job ? (
        <TransferProgress job={job}>
          <span>{connection_text}</span>
        </TransferProgress>
      ) : (
        <>
          <progress
            aria-label="Job progress"
            max={1}
            value={job.progress ?? undefined}
          />
          <div className="progress_labels">
            <span>
              {global_step == null
                ? job.progress == null
                  ? 'Waiting for progress'
                  : `${Math.round(job.progress * 100)}% complete`
                : `Step ${global_step}`}
            </span>
            <span>{connection_text}</span>
          </div>
        </>
      )}
      <div className="monitor_details">
        <div>
          <span>Last update</span>
          <strong>{new Date(job.updated_at).toLocaleString()}</strong>
        </div>
        {job.operation === 'train' && (
          <div>
            <span>Automatic saves</span>
            <strong>Every 5 minutes</strong>
          </div>
        )}
        {fetch_job && (
          <div>
            <span>Files</span>
            <strong>{file_progress_text(job.metrics)}</strong>
          </div>
        )}
        <div>
          <span>Time remaining</span>
          <strong>
            {job.estimated_seconds_remaining == null
              ? 'Not estimated'
              : `${Math.ceil(job.estimated_seconds_remaining / 60)} min`}
          </strong>
        </div>
      </div>
      {pending_action && confirmed && (
        <p className="notice" role="status">
          {requested_action_text(job, fetch_job)}
        </p>
      )}
      {job.error && (
        <div className="error_notice" role="alert">
          {job.error.message ||
            'This job could not finish. Check the saved result before retrying.'}
        </div>
      )}
      {job.status === 'completed' && !fetch_job && (
        <p className="help_text">
          Work finished. This does not mean the model is approved for Encode or
          Decode.
        </p>
      )}
      {!confirmed && (
        <p className="notice">
          Job actions are unavailable until the backend confirms the current
          status.
        </p>
      )}
      <EvaluationResultCard job={job} />
      <DatasetSummaryCard job={job} />
      <ErrorNotice error={action.error} />
      <div className="form_actions">
        {uncertain && attempt ? (
          <button
            className="button secondary"
            disabled={!confirmed}
            onClick={() => action.mutate(attempt)}
          >
            Retry job action
          </button>
        ) : (
          job.available_actions
            .filter((kind): kind is ActionKind => kind in labels)
            .filter(() => !(fetch_job && pending_action))
            .map((kind) => (
              <button
                key={kind}
                className={`button ${kind === 'stop' ? 'primary' : 'secondary'}`}
                disabled={!confirmed || action.isPending || pending_action}
                onClick={() => request_action(kind)}
              >
                {labels[kind]}
              </button>
            ))
        )}
      </div>
      <p className="help_text">
        {fetch_job
          ? 'The download keeps running when you switch tabs or close this page.'
          : 'Training keeps running when you switch tabs or close this page.'}
      </p>
    </section>
  );
}
