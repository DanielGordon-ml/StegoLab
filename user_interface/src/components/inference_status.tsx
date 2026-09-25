import { useMutation, useQueryClient } from '@tanstack/react-query';
import { act_on_job } from '../contracts/workflow_service';
import type { JobSnapshot } from '../contracts/workflows';
import { ErrorNotice } from './error_notice';

/** Say in plain words what one encode or decode job is doing right now. */
export function describe_job(
  job: JobSnapshot,
  operation: 'encode' | 'decode',
  training_active: boolean,
) {
  const verb = operation === 'encode' ? 'Encoding' : 'Decoding';
  if (job.status === 'queued')
    return training_active
      ? `Queued behind a training job. ${verb} starts when that job finishes.`
      : 'Queued. Waiting for the local worker.';
  if (job.status === 'running')
    return job.phase === 'verifying_saved_image'
      ? 'Verifying the saved image with the matching decoder.'
      : 'Processing.';
  if (job.status === 'completed')
    return operation === 'encode'
      ? 'The saved PNG passed verification.'
      : 'The message was recovered.';
  if (job.status === 'cancelled') return 'Cancelled before it started.';
  return job.error?.message ?? 'This job did not finish.';
}

const labels: Record<JobSnapshot['status'], string> = {
  queued: 'Queued',
  running: 'Processing',
  paused: 'Paused',
  stopped: 'Stopped',
  completed: 'Ready',
  cancelled: 'Cancelled',
  failed: 'Failed',
  interrupted: 'Interrupted',
  needs_input: 'Needs input',
};

/** Show the job phase, its plain error, and the cancel action while queued. */
export function InferenceStatus({
  job,
  operation,
  training_active,
}: {
  job: JobSnapshot;
  operation: 'encode' | 'decode';
  training_active: boolean;
}) {
  const client = useQueryClient();
  const cancel = useMutation({
    mutationFn: () =>
      act_on_job(job.job_identifier, {
        client_request_identifier: crypto.randomUUID(),
        action: 'cancel',
      }),
    onSuccess: () => void client.invalidateQueries({ queryKey: ['jobs'] }),
  });
  const label =
    job.status === 'running' && job.phase === 'verifying_saved_image'
      ? 'Verifying'
      : labels[job.status];
  return (
    <div className="inference_status">
      <p role="status">
        <span className={`status_pill ${job.status}`}>{label}</span>{' '}
        {describe_job(job, operation, training_active)}
      </p>
      {job.available_actions.includes('cancel') && (
        <button
          className="button secondary"
          disabled={cancel.isPending}
          onClick={() => cancel.mutate()}
        >
          Cancel job
        </button>
      )}
      <ErrorNotice error={cancel.error} />
    </div>
  );
}
