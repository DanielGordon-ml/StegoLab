import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DatasetSourceUpload } from '../components/dataset_source_upload';
import { NAME_CHANGED_TEXT } from '../components/dataset_upload_stages';
import type { Capabilities } from '../contracts/capabilities';
import type { DatasetInspection } from '../contracts/datasets';
import { UPLOAD_MEMORY_KEY } from '../contracts/dataset_upload_flow';
import { FakeRequest } from './fake_request';
import { capabilities } from './fixtures';

const SIZE = 1024;
const LAST_MODIFIED = 1700000000000;
const identifier = `upload_${'0'.repeat(31)}1`;
const archive = new File([new Uint8Array(SIZE)], 'photos.zip', {
  type: 'application/zip',
  lastModified: LAST_MODIFIED,
});
const memory = {
  identifier,
  file_name: 'photos.zip',
  size: SIZE,
  last_modified: LAST_MODIFIED,
};
/** Raw folder names the stub backend reports as holding different files. */
const TAKEN_NAMES = ['taken'];

/** Session record with every field the response contract requires. */
function session(received: number[], complete = false) {
  return {
    upload_identifier: identifier,
    file_name: 'photos.zip',
    total_bytes: SIZE,
    chunk_bytes: 16777216,
    chunk_count: 1,
    received_chunks: received,
    complete,
    sha256: null,
    created_at: '2026-09-27T10:00:00Z',
    expires_at: '2026-09-28T10:00:00Z',
  };
}
/** What the backend reports for a small archive of two images. */
const inspection: DatasetInspection = {
  source_kind: 'upload',
  reference: 'photos.zip',
  requested_revision: null,
  resolved_revision: null,
  suggested_source_name: 'photos',
  access: 'available',
  access_guidance: null,
  content: 'images',
  asset_count: 2,
  assets: [],
  download_bytes: SIZE,
  materialized_bytes: 2048,
  cached_bytes: 0,
  supports_pause: false,
  declared_splits: false,
  materialization_identity: 'a'.repeat(64),
  raw_folder: 'available',
  free_disk_bytes: 50000000000,
  required_free_bytes: 4096,
  disk_sufficient: true,
  server_token_configured: false,
  warnings: [],
};
const not_found = () =>
  Response.json(
    {
      error: {
        code: 'upload_expired',
        message: 'This upload expired. Start it again.',
        diagnostic_reference: '0123456789abcdef0123456789abcdef',
      },
    },
    { status: 404 },
  );
const backend = { read: not_found };
/** Raw folder names sent with each inspection, in order. */
const checked_names: (string | null)[] = [];

beforeEach(() => {
  backend.read = not_found;
  checked_names.length = 0;
  sessionStorage.clear();
  FakeRequest.reset();
  vi.stubGlobal('XMLHttpRequest', FakeRequest);
  vi.stubGlobal('crypto', {
    randomUUID: () => 'request-1',
    subtle: { digest: async () => new Uint8Array(32).buffer },
  });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, options?: RequestInit) => {
      const method = options?.method ?? 'GET';
      if (method === 'DELETE') return new Response(null, { status: 204 });
      if (method === 'GET') return backend.read();
      if (url.endsWith('/complete')) return Response.json(session([0], true));
      if (url.endsWith('/datasets/uploads'))
        return Response.json(session([]), { status: 201 });
      const request = JSON.parse(String(options?.body)) as {
        source_name: string | null;
      };
      checked_names.push(request.source_name);
      const taken = TAKEN_NAMES.includes(request.source_name ?? '');
      return Response.json({
        ...inspection,
        raw_folder: taken ? 'conflict' : 'available',
      });
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

/** Mount the upload form with a fresh query cache and no retries. */
function render_form() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const view = render(
    <QueryClientProvider client={client}>
      <DatasetSourceUpload
        capabilities={capabilities as Capabilities}
        on_started={vi.fn()}
      />
    </QueryClientProvider>,
  );
  return { user: userEvent.setup(), container: view.container };
}
/** Answer the single part the way the backend would, with a matching digest. */
async function answer_part() {
  const part = await FakeRequest.next_send();
  part.respond(200, {
    upload_identifier: identifier,
    index: 0,
    bytes: SIZE,
    sha256: '0'.repeat(64),
    received_chunks: [0],
  });
}
/** Choose the archive, answer its part and wait for the automatic check. */
async function upload_archive(
  user: ReturnType<typeof userEvent.setup>,
  folder: string,
) {
  await user.upload(screen.getByLabelText('Archive file'), archive);
  await answer_part();
  expect(await screen.findByText(`${folder} will be created`)).toBeVisible();
}
const fetch_button = () =>
  screen.getByRole('button', { name: 'Fetch and prepare' });
const check_button = () =>
  screen.getByRole('button', { name: 'Check the archive again' });
const name_field = () => screen.getByLabelText('Save raw files as');
const focused = () => document.activeElement;

describe('dataset archive upload form', () => {
  it('checks a renamed raw folder again before it allows the fetch', async () => {
    const { user } = render_form();
    await upload_archive(user, 'data/photos');
    expect(checked_names).toEqual([null]);
    expect(name_field()).toHaveValue('photos');
    expect(fetch_button()).toBeEnabled();
    await user.clear(name_field());
    await user.type(name_field(), 'taken');
    expect(fetch_button()).toBeDisabled();
    expect(screen.getByText(NAME_CHANGED_TEXT)).toBeVisible();
    expect(screen.queryByText('data/photos will be created')).toBeNull();
    await user.click(check_button());
    expect(
      await screen.findByText(
        'data/taken already holds different files — choose another name',
      ),
    ).toBeVisible();
    expect(checked_names).toEqual([null, 'taken']);
    expect(fetch_button()).toBeDisabled();
    await user.clear(name_field());
    await user.type(name_field(), 'photos_2');
    await user.click(check_button());
    expect(
      await screen.findByText('data/photos_2 will be created'),
    ).toBeVisible();
    expect(fetch_button()).toBeEnabled();
    // A later archive is checked for the name that is already typed.
    await user.click(
      screen.getByRole('button', { name: 'Choose a different archive' }),
    );
    expect(focused()).toBe(screen.getByLabelText('Archive file'));
    await upload_archive(user, 'data/photos_2');
    expect(checked_names).toEqual([null, 'taken', 'photos_2', 'photos_2']);
  });
  it('drops the resume notice once the remembered upload has finished', async () => {
    sessionStorage.setItem(UPLOAD_MEMORY_KEY, JSON.stringify(memory));
    backend.read = () => Response.json(session([]));
    const { user, container } = render_form();
    expect(
      screen.getByText(/An earlier upload of photos.zip was not finished/),
    ).toBeVisible();
    await user.upload(screen.getByLabelText('Archive file'), archive);
    expect(focused()).toBe(container.querySelector('.upload_stage'));
    expect(
      await screen.findByText(/Continuing from part 1 of 1\./),
    ).toBeVisible();
    await answer_part();
    expect(
      await screen.findByText('data/photos will be created'),
    ).toBeVisible();
    await user.click(
      screen.getByRole('button', { name: 'Choose a different archive' }),
    );
    expect(screen.getByLabelText('Archive file')).toBeVisible();
    expect(screen.queryByText(/was not finished/)).toBeNull();
  });
  it('keeps the keyboard inside the form when the upload is cancelled', async () => {
    const { user, container } = render_form();
    await user.upload(screen.getByLabelText('Archive file'), archive);
    await FakeRequest.next_send();
    await user.click(screen.getByRole('button', { name: 'Cancel upload' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The upload was cancelled.',
    );
    expect(screen.getByLabelText('Archive file')).toBeVisible();
    expect(focused()).toBe(container.querySelector('.upload_stage'));
  });
});
