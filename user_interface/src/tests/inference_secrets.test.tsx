import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Application } from '../application';
import { validate_decoding_request } from '../contracts/validation';
import { discarded_texts } from '../hooks/use_decoded_text';
import type { JobSnapshot } from '../contracts/workflows';
import {
  completed_decode_job,
  decode_job,
  encoded_upload,
  mock_inference_backend,
} from './inference_fixtures';

vi.mock('../contracts/upload', () => ({
  upload_image: vi.fn(async () => encoded_upload),
}));

/** Mount the shell and expose its cache so tests can inspect what it holds. */
function render_decode() {
  vi.stubGlobal('URL', {
    ...URL,
    createObjectURL: vi.fn(() => 'blob:preview'),
    revokeObjectURL: vi.fn(),
  });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <Application />
    </QueryClientProvider>,
  );
  return { user: userEvent.setup(), client };
}

/** Serialize every cached query and mutation so a secret can be searched for. */
function cache_contents(client: QueryClient) {
  return JSON.stringify({
    queries: client
      .getQueryCache()
      .getAll()
      .map((query) => query.state.data),
    mutations: client
      .getMutationCache()
      .getAll()
      .map((mutation) => mutation.state),
  });
}

const encoded = new File(['png-bytes'], 'encoded.png', { type: 'image/png' });
beforeEach(() => discarded_texts.clear());
afterEach(() => vi.unstubAllGlobals());

describe('decode secrets', () => {
  it('sends the password once, clears the field, and keeps text out of the cache', async () => {
    let jobs: JobSnapshot[] = [];
    const fetch = mock_inference_backend({ jobs: () => jobs });
    const { user, client } = render_decode();
    await user.click(screen.getByRole('tab', { name: 'Decode' }));
    await user.upload(await screen.findByLabelText('Encoded PNG'), encoded);
    await screen.findByText(/1024×768 px/);
    const password = screen.getByLabelText('Password');
    await user.type(password, 'sentinel-password');
    await user.click(
      screen.getByRole('button', { name: 'Authenticate and decode' }),
    );
    expect(
      await screen.findByText('Queued. Waiting for the local worker.'),
    ).toBeVisible();
    expect(password).toHaveValue('');
    const submissions = fetch.mock.calls.filter(([url]) =>
      String(url).endsWith('/decoding_jobs'),
    );
    expect(submissions).toHaveLength(1);
    const body = JSON.parse(String(submissions[0][1]?.body));
    expect(validate_decoding_request(body)).toBe(true);
    expect(body.password).toBe('sentinel-password');
    await waitFor(() =>
      expect(cache_contents(client)).not.toContain('sentinel-password'),
    );
    jobs = [completed_decode_job];
    client.setQueryData(['jobs'], { items: [completed_decode_job] });
    const recovered = await screen.findByLabelText('Recovered message');
    await waitFor(() => expect(recovered).toHaveValue('hello'));
    expect(cache_contents(client)).not.toContain('hello');
    await user.click(screen.getByRole('button', { name: 'Clear message' }));
    expect(recovered).toHaveValue('');
    const deletions = () =>
      fetch.mock.calls.filter(([, options]) => options?.method === 'DELETE');
    await waitFor(() => expect(deletions()).toHaveLength(1));
    expect(String(deletions()[0][0])).toContain(
      '/jobs/job_decode/decoded_text',
    );
  });
  it('forgets loaded text when leaving Decode and explains an expired result', async () => {
    const fetch = mock_inference_backend({
      jobs: () => [completed_decode_job],
    });
    const { user, client } = render_decode();
    await user.click(screen.getByRole('tab', { name: 'Decode' }));
    const recovered = await screen.findByLabelText('Recovered message');
    await waitFor(() => expect(recovered).toHaveValue('hello'));
    await user.click(screen.getByRole('tab', { name: 'Encode' }));
    await waitFor(() =>
      expect(
        fetch.mock.calls.filter(([, options]) => options?.method === 'DELETE'),
      ).toHaveLength(1),
    );
    await user.click(screen.getByRole('tab', { name: 'Decode' }));
    expect(screen.getByLabelText('Recovered message')).toHaveValue('');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(
      fetch.mock.calls.filter(
        ([url, options]) =>
          String(url).endsWith('/decoded_text') && options?.method !== 'DELETE',
      ),
    ).toHaveLength(1);
    await user.click(screen.getByRole('tab', { name: 'Encode' }));
    const never_loaded = {
      ...completed_decode_job,
      job_identifier: 'job_decode_later',
      created_at: '2026-09-24T11:00:00Z',
    };
    client.setQueryData(['jobs'], {
      items: [completed_decode_job, never_loaded],
    });
    mock_inference_backend({
      jobs: () => [completed_decode_job, never_loaded],
      text: () =>
        Response.json(
          {
            error: {
              code: 'result_expired',
              message:
                'The recovered text expired. Enter the password and decode again.',
              diagnostic_reference: '0123456789abcdef0123456789abcdef',
            },
          },
          { status: 404 },
        ),
    });
    await user.click(screen.getByRole('tab', { name: 'Decode' }));
    expect(
      await screen.findByText(
        'The recovered text expired. Enter the password and decode again.',
      ),
    ).toBeVisible();
    expect(screen.getByRole('button', { name: 'Copy message' })).toBeDisabled();
    expect(screen.getByLabelText('Recovered message')).toHaveValue('');
    expect(decode_job.status).toBe('queued');
  });
});
