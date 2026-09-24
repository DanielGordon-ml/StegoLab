import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { TrainingSetup } from '../components/training_setup';
import {
  prepare_request,
  validate_workflow,
  validate_workspace,
  validate_job,
} from '../contracts/workflow_service';
import {
  empty_workspace,
  eligible_dataset,
  saved_job,
} from './workflow_fixtures';
import type { WorkspaceCheckpoint } from '../contracts/workspace';

const workspace = { ...empty_workspace, datasets: [eligible_dataset] };

/** Exercise the real wizard with strict exported schemas and a fresh cache. */
function render_setup(resume?: WorkspaceCheckpoint) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const on_started = vi.fn();
  render(
    <QueryClientProvider client={client}>
      <TrainingSetup
        workspace={workspace}
        resume={resume}
        on_close={vi.fn()}
        on_started={on_started}
      />
    </QueryClientProvider>,
  );
  return { user: userEvent.setup(), on_started };
}

/** Fill only user-editable settings, leaving the immutable profile unchanged. */
async function review_setup(user: ReturnType<typeof userEvent.setup>) {
  await user.selectOptions(
    screen.getByLabelText('Prepared dataset'),
    'dataset_example',
  );
  await user.click(screen.getByRole('button', { name: /Continue to setup/ }));
  await user.type(screen.getByLabelText('Run name'), 'example');
  await user.click(screen.getByRole('button', { name: /Check and review/ }));
}

describe('training requests', () => {
  it('fills backend-owned defaults before applying the complete exported schema', () => {
    const request = prepare_request({
      operation: 'train',
      client_request_identifier: 'request_one',
      dataset_identifier: 'dataset_example',
    });
    expect(validate_workflow(request)).toBe(true);
    expect(request).toMatchObject({
      cpu_threads: 4,
      stop_after_step: 1000,
      checkpoint_identifier: null,
      lightweight: false,
    });
    expect(() => prepare_request({ ...request, cpu_threads: 0 })).toThrow(
      'Check the run settings',
    );
    expect(validate_workspace(workspace)).toBe(true);
    expect(validate_job(saved_job)).toBe(true);
  });
  it('checks compatibility without starting work and freezes the start request for uncertain retries', async () => {
    let starts = 0;
    const fetch = vi.fn(async (url: string, options?: RequestInit) => {
      if (url.endsWith('/training_preflight'))
        return Response.json({
          allowed: true,
          blockers: [],
          warnings: [],
          resolved_settings: {},
          remaining_budget_seconds: 500,
          remaining_experiment_slots: 1,
        });
      starts += 1;
      expect(validate_workflow(JSON.parse(String(options?.body)))).toBe(true);
      if (starts === 1) throw new Error('offline');
      return Response.json(saved_job);
    });
    vi.stubGlobal('fetch', fetch);
    const { user, on_started } = render_setup();
    await review_setup(user);
    expect(await screen.findByText('Ready to start')).toBeVisible();
    expect(starts).toBe(0);
    await user.click(screen.getByRole('button', { name: 'Start training' }));
    const retry = await screen.findByRole('button', { name: 'Retry start' });
    expect(screen.getByRole('button', { name: 'Back' })).toBeDisabled();
    await user.click(retry);
    expect(on_started).toHaveBeenCalledWith(saved_job);
    const writes = fetch.mock.calls.filter(([url]) =>
      url.endsWith('/training_jobs'),
    );
    expect(writes[0][1]?.body).toBe(writes[1][1]?.body);
  });
  it('keeps compatibility blockers visible and prevents start', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        Response.json({
          allowed: false,
          blockers: ['The original experiment budget is exhausted.'],
          warnings: [],
          resolved_settings: {},
          remaining_budget_seconds: 0,
          remaining_experiment_slots: 0,
        }),
      ),
    );
    const { user } = render_setup();
    await review_setup(user);
    expect(
      await screen.findByText('The original experiment budget is exhausted.'),
    ).toBeVisible();
    expect(
      screen.getByRole('button', { name: 'Start training' }),
    ).toBeDisabled();
  });
  it('restores and locks the original dataset and threads when resuming', async () => {
    const resume: WorkspaceCheckpoint = {
      identifier: 'checkpoint_one',
      experiment_identifier: 'original_run',
      dataset_revision: 'revision_1',
      cpu_threads: 8,
      global_step: 10,
      created_at: saved_job.created_at,
      latest: true,
      pinned: false,
      metrics: {
        exact_recovery_rate: null,
        trial_count: null,
        psnr: null,
        psnr_sample_count: null,
        ssim: null,
        ssim_sample_count: null,
      },
      resume_blockers: [],
    };
    const { user } = render_setup(resume);
    expect(screen.getByLabelText('Prepared dataset')).toHaveValue(
      'dataset_example',
    );
    expect(screen.getByLabelText('Prepared dataset')).toBeDisabled();
    await user.click(screen.getByRole('button', { name: /Continue to setup/ }));
    expect(screen.getByLabelText('CPU threads')).toHaveValue(8);
    expect(screen.getByLabelText('CPU threads')).toHaveAttribute('readonly');
    expect(screen.getByLabelText('Run name')).toHaveValue('original_run');
  });
});
