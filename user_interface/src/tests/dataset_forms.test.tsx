import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DatasetPreparation } from '../components/dataset_preparation';
import {
  TOKEN_CONFIGURED_TEXT,
  TOKEN_MISSING_TEXT,
} from '../components/dataset_source_hugging_face';
import { CHECKSUM_RULE, HTTPS_RULE } from '../components/dataset_source_https';
import type { Capabilities } from '../contracts/capabilities';
import type { DatasetInspection } from '../contracts/datasets';
import {
  validate_fetch_request,
  validate_inspection_request,
} from '../contracts/dataset_service';
import type { JobSnapshot } from '../contracts/workflows';
import { capabilities } from './fixtures';
import { empty_workspace, saved_job } from './workflow_fixtures';

const all_sources: Capabilities = {
  ...(capabilities as Capabilities),
  dataset_source_kinds: [
    'server_folder',
    'upload',
    'hugging_face',
    'https_archive',
  ],
};
const GUIDANCE =
  'Accept the dataset terms on huggingface.co and add a token on the server.';

/** What the backend reports for a public repository that can be fetched. */
const inspection: DatasetInspection = {
  source_kind: 'hugging_face',
  reference: 'example/street_photos',
  requested_revision: 'main',
  resolved_revision: '0123456789abcdef0123456789abcdef01234567',
  suggested_source_name: 'example_street_photos',
  access: 'available',
  access_guidance: null,
  content: 'images',
  asset_count: 12,
  assets: [],
  download_bytes: 1200000000,
  materialized_bytes: 1500000000,
  cached_bytes: 0,
  supports_pause: true,
  declared_splits: false,
  materialization_identity: 'a'.repeat(64),
  raw_folder: 'available',
  free_disk_bytes: 50000000000,
  required_free_bytes: 3000000000,
  disk_sufficient: true,
  server_token_configured: false,
  warnings: [],
};
const fetch_job: JobSnapshot = {
  ...saved_job,
  job_identifier: 'job_fetch',
  operation: 'fetch_dataset',
  experiment_identifier: null,
  phase: 'resolving',
};

interface Backend {
  inspect?: () => Response;
  start?: () => Response;
}

/** Answer inspection and fetch requests with contract shapes and record them. */
function stub_backend(backend: Backend = {}) {
  const calls: { url: string; body: string }[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, options?: RequestInit) => {
      calls.push({ url, body: String(options?.body ?? '') });
      if (url.endsWith('/datasets/inspections'))
        return backend.inspect?.() ?? Response.json(inspection);
      if (url.endsWith('/datasets/fetch_jobs'))
        return backend.start?.() ?? Response.json(fetch_job, { status: 202 });
      return Response.json({ items: [] });
    }),
  );
  return calls;
}

/** Render the open dataset card with a fresh cache and the given capabilities. */
function render_card(overrides: Partial<Capabilities> = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const on_started = vi.fn();
  const view = render(
    <QueryClientProvider client={client}>
      <DatasetPreparation
        workspace={empty_workspace}
        capabilities={{ ...all_sources, ...overrides }}
        on_started={on_started}
      />
    </QueryClientProvider>,
  );
  return { user: userEvent.setup(), on_started, unmount: view.unmount };
}

async function choose_hugging_face(
  user: ReturnType<typeof userEvent.setup>,
  repository = 'example/street_photos',
) {
  await user.click(screen.getByText('Prepare a dataset'));
  await user.click(screen.getByRole('radio', { name: 'Hugging Face' }));
  await user.type(screen.getByLabelText('Repository'), repository);
}
const check_button = () => screen.getByRole('button', { name: 'Check source' });
const fetch_button = () =>
  screen.getByRole('button', { name: 'Fetch and prepare' });
const start_calls = (calls: { url: string; body: string }[]) =>
  calls.filter((call) => call.url.endsWith('/datasets/fetch_jobs'));

afterEach(() => vi.unstubAllGlobals());

describe('dataset source forms', () => {
  it('disables the source check until the repository identifier is valid', async () => {
    stub_backend();
    const { user } = render_card();
    await choose_hugging_face(user, 'not-a-repository');
    expect(check_button()).toBeDisabled();
    expect(screen.getByText(/Use the form owner\/name/)).toBeInTheDocument();
    await user.type(screen.getByLabelText('Repository'), '/photos');
    expect(check_button()).toBeEnabled();
  });
  it('tells whether the server holds a Hugging Face token and which kinds are enabled', async () => {
    stub_backend();
    const configured = render_card({ hugging_face_token_configured: true });
    await choose_hugging_face(configured.user);
    expect(screen.getByText(TOKEN_CONFIGURED_TEXT)).toBeInTheDocument();
    configured.unmount();
    const missing = render_card({ hugging_face_token_configured: false });
    await choose_hugging_face(missing.user);
    expect(screen.getByText(TOKEN_MISSING_TEXT)).toBeInTheDocument();
    missing.unmount();
    render_card({ dataset_source_kinds: ['server_folder'] });
    expect(screen.getByRole('radio', { name: 'Hugging Face' })).toBeDisabled();
    expect(screen.getAllByText('Not enabled on this server')).toHaveLength(3);
  });
  it('shows access guidance for a gated repository and keeps fetching disabled', async () => {
    stub_backend({
      inspect: () =>
        Response.json({
          ...inspection,
          access: 'access_required',
          access_guidance: GUIDANCE,
        }),
    });
    const { user } = render_card();
    await choose_hugging_face(user);
    await user.click(check_button());
    expect(await screen.findByText(GUIDANCE)).toBeInTheDocument();
    expect(fetch_button()).toBeDisabled();
  });
  it('keeps fetching disabled when the raw folder holds different files', async () => {
    stub_backend({
      inspect: () => Response.json({ ...inspection, raw_folder: 'conflict' }),
    });
    const { user } = render_card();
    await choose_hugging_face(user);
    await user.type(screen.getByLabelText('Save raw files as'), 'street');
    await user.click(check_button());
    expect(
      await screen.findByText(
        'data/street already holds different files — choose another name',
      ),
    ).toBeInTheDocument();
    expect(fetch_button()).toBeDisabled();
  });
  it('rejects a plain http address and a malformed checksum in the browser', async () => {
    stub_backend();
    const { user } = render_card();
    await user.click(screen.getByText('Prepare a dataset'));
    await user.click(screen.getByRole('radio', { name: 'HTTPS archive' }));
    const address = screen.getByLabelText('Archive address');
    await user.type(address, 'http://example.org/images.zip');
    expect(screen.getByText(HTTPS_RULE)).toBeInTheDocument();
    expect(check_button()).toBeDisabled();
    await user.clear(address);
    await user.type(address, 'https://example.org/images.zip');
    expect(screen.queryByText(HTTPS_RULE)).toBeNull();
    expect(check_button()).toBeEnabled();
    await user.type(
      screen.getByLabelText('Expected SHA-256 (optional)'),
      'not-a-digest',
    );
    expect(screen.getByText(CHECKSUM_RULE)).toBeInTheDocument();
    expect(check_button()).toBeDisabled();
  });
  it('starts a fetch job from the checked details with a body the contract accepts', async () => {
    const calls = stub_backend();
    const { user, on_started } = render_card();
    await choose_hugging_face(user);
    await user.click(check_button());
    expect(
      await screen.findByText('Download size: 1.2 GB'),
    ).toBeInTheDocument();
    expect(
      screen.getByText('Free space: 50.0 GB (enough)'),
    ).toBeInTheDocument();
    expect(screen.getByText('Pause: available')).toBeInTheDocument();
    expect(screen.getByLabelText('Save raw files as')).toHaveValue(
      'example_street_photos',
    );
    const inspect_call = calls.find((call) =>
      call.url.endsWith('/datasets/inspections'),
    );
    expect(
      validate_inspection_request(JSON.parse(inspect_call?.body ?? '')),
    ).toBe(true);
    await user.click(fetch_button());
    await waitFor(() => expect(on_started).toHaveBeenCalledWith(fetch_job));
    const body = JSON.parse(start_calls(calls)[0].body);
    expect(validate_fetch_request(body)).toBe(true);
    expect(body).toMatchObject({
      operation: 'fetch_dataset',
      source_name: 'example_street_photos',
      dataset_name: 'example_street_photos',
      maximum_images: 200000,
      prepare: true,
      source: {
        source_kind: 'hugging_face',
        repository: 'example/street_photos',
        revision: 'main',
        content: 'images',
      },
    });
  });
  it('offers a retry after a network failure and re-sends the identical request', async () => {
    let attempts = 0;
    const calls = stub_backend({
      start: () => {
        attempts += 1;
        if (attempts === 1) throw new Error('offline');
        return Response.json(fetch_job, { status: 202 });
      },
    });
    const { user, on_started } = render_card();
    await choose_hugging_face(user);
    await user.click(check_button());
    await screen.findByText('Download size: 1.2 GB');
    await user.click(fetch_button());
    await user.click(
      await screen.findByRole('button', { name: 'Retry fetch' }),
    );
    await waitFor(() => expect(on_started).toHaveBeenCalledWith(fetch_job));
    const starts = start_calls(calls);
    expect(starts).toHaveLength(2);
    expect(starts[0].body).toBe(starts[1].body);
  });
});
