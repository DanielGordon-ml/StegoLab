import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, expect, it } from 'vitest';
import { JobMonitor } from '../components/job_monitor';
import { phase_label } from '../components/job_transfer';
import type { JobSnapshot } from '../contracts/workflows';
import { fetch_job } from './workflow_fixtures';

/** Mount the monitor with a fresh query cache and a live event stream. */
function render_monitor(job: JobSnapshot) {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <JobMonitor job={job} stream_connected={true} />
    </QueryClientProvider>,
  );
}

describe('dataset download monitor', () => {
  it('shows received bytes against a known total', () => {
    render_monitor(fetch_job);
    const bar = screen.getByRole('progressbar', { name: 'Download progress' });
    expect(bar).toHaveAttribute('max', '100000000');
    expect(bar).toHaveAttribute('value', '12000000');
    expect(screen.getByText('12.0 of 100.0 MB')).toBeVisible();
    expect(screen.getByText('Downloading')).toBeVisible();
    expect(screen.getByText('Dataset download')).toBeVisible();
    expect(
      screen.getByRole('button', { name: 'Pause download' }),
    ).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Cancel job' })).toBeEnabled();
  });
  it('shows activity and the received amount while the total is unknown', () => {
    const metrics = { ...fetch_job.metrics };
    delete metrics.bytes_total;
    render_monitor({ ...fetch_job, progress: null, metrics });
    const bar = screen.getByRole('progressbar', { name: 'Download progress' });
    expect(bar).not.toHaveAttribute('value');
    expect(bar).toHaveClass('activity_indicator');
    const status = screen.getByText(/received so far/);
    expect(status).toHaveTextContent('Working… 12.0 MB received so far');
    expect(status).toHaveAttribute('role', 'status');
  });
  it('reports a cancel the backend has not confirmed and hides the actions', () => {
    render_monitor({
      ...fetch_job,
      phase: 'cancelling',
      requested_action: 'cancel',
      available_actions: ['cancel'],
    });
    const notice = screen.getByText(
      'Cancelling and cleaning up. Waiting for the backend to confirm.',
    );
    expect(notice).toHaveAttribute('role', 'status');
    expect(screen.getByText('Cancelling')).toBeVisible();
    expect(screen.queryAllByRole('button')).toHaveLength(0);
  });
  it('offers resume and cancel for a paused download', () => {
    render_monitor({
      ...fetch_job,
      status: 'paused',
      phase: 'paused',
      available_actions: ['resume', 'cancel'],
    });
    expect(
      screen.getByRole('button', { name: 'Resume download' }),
    ).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Cancel job' })).toBeEnabled();
    expect(
      screen.queryByRole('button', { name: 'Pause download' }),
    ).not.toBeInTheDocument();
    expect(screen.getByText('12.0 of 100.0 MB')).toBeVisible();
  });
});

it('turns download phases into plain labels', () => {
  expect(phase_label('resolving')).toBe('Resolving source');
  expect(phase_label('preparing')).toBe('Preparing images');
  expect(phase_label('cleaning')).toBe('Cleaning up');
  expect(phase_label('waiting_for_worker')).toBe('waiting for worker');
});
