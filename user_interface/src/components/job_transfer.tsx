import type { ReactNode } from 'react';
import type { JobSnapshot } from '../contracts/workflows';
import { format_byte_progress, format_bytes } from './byte_format';

const PHASE_LABELS: Record<string, string> = {
  resolving: 'Resolving source',
  downloading: 'Downloading',
  extracting: 'Extracting',
  preparing: 'Preparing images',
  pausing: 'Pausing',
  paused: 'Paused',
  cancelling: 'Cancelling',
  cleaning: 'Cleaning up',
  completed: 'Completed',
  cancelled: 'Cancelled',
};

/** Describe a dataset download phase in plain words; unknown phases keep their name. */
export function phase_label(phase: string): string {
  return PHASE_LABELS[phase] ?? phase.replaceAll('_', ' ');
}

/** Show received bytes against the total, or activity only while the total is unknown. */
export function TransferProgress({
  job,
  children,
}: {
  job: JobSnapshot;
  children?: ReactNode;
}) {
  const received: number | undefined = job.metrics.bytes_received;
  const total: number | undefined = job.metrics.bytes_total;
  const active = job.status === 'queued' || job.status === 'running';
  if (total != null && total > 0)
    return (
      <div className="transfer_progress">
        <progress
          aria-label="Download progress"
          max={total}
          value={Math.min(received ?? 0, total)}
        />
        <div className="progress_labels">
          <span>{format_byte_progress(received ?? 0, total)}</span>
          {children}
        </div>
      </div>
    );
  return (
    <div className="transfer_progress">
      <progress
        aria-label="Download progress"
        className={active ? 'activity_indicator' : undefined}
        max={1}
        value={active ? undefined : job.status === 'completed' ? 1 : 0}
      />
      <div className="progress_labels">
        <span role="status">
          {active ? 'Working… ' : ''}
          {format_bytes(received ?? 0)} received so far
        </span>
        {children}
      </div>
    </div>
  );
}
