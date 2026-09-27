import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DatasetStorageCard } from '../components/dataset_storage_card';
import { dataset_storage } from './fixtures';

/** Mount the card with a fresh query cache and no retries. */
function render_card() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <DatasetStorageCard />
    </QueryClientProvider>,
  );
  return userEvent.setup();
}

/** Answer the storage read, and the removal when a handler is given. */
function stub_backend(read: object, remove?: () => Response) {
  const fetch = vi.fn(async (url: string | URL, options?: RequestInit) =>
    options?.method === 'DELETE' && remove ? remove() : Response.json(read),
  );
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
afterEach(() => vi.unstubAllGlobals());

describe('dataset storage card', () => {
  it('shows every storage row with readable sizes', async () => {
    stub_backend(dataset_storage);
    render_card();
    expect(
      await screen.findByText('3 entries · 1.5 GB (500.0 MB unused)'),
    ).toBeVisible();
    expect(screen.getByText('2 · 1.2 GB')).toBeVisible();
    expect(screen.getByText('4 revisions · 800.0 MB')).toBeVisible();
    expect(
      screen.getByText('40.0 GB (keeps at least 5.0 GB free)'),
    ).toBeVisible();
    expect(
      screen.getByRole('button', { name: 'Remove unused downloads' }),
    ).toBeEnabled();
  });
  it('keeps removal unavailable while a download is running', async () => {
    stub_backend({ ...dataset_storage, active_fetch_jobs: 1 });
    render_card();
    expect(
      await screen.findByRole('button', { name: 'Remove unused downloads' }),
    ).toBeDisabled();
    expect(
      screen.getByText(
        'Removal waits until the running dataset download has finished.',
      ),
    ).toBeVisible();
  });
  it('asks for confirmation, then deletes and shows the refreshed numbers', async () => {
    const fetch = stub_backend(dataset_storage, () =>
      Response.json({
        ...dataset_storage,
        cache_bytes: 1000000000,
        cache_entries: 1,
        cache_unused_bytes: 0,
        cache_unused_entries: 0,
        cleanup_available: false,
      }),
    );
    const user = render_card();
    await user.click(
      await screen.findByRole('button', { name: 'Remove unused downloads' }),
    );
    const deletes = () =>
      fetch.mock.calls.filter(([, options]) => options?.method === 'DELETE');
    expect(deletes()).toHaveLength(0);
    await user.click(screen.getByRole('button', { name: 'Confirm removal' }));
    expect(
      await screen.findByText('1 entry · 1.0 GB (0 B unused)'),
    ).toBeVisible();
    expect(deletes()).toHaveLength(1);
    expect(String(deletes()[0][0])).toBe('/api/v1/datasets/cache/unused');
    expect(screen.getByText('Unused downloads were removed.')).toBeVisible();
    expect(
      screen.getByRole('button', { name: 'Remove unused downloads' }),
    ).toBeDisabled();
    expect(
      screen.getByText('There are no unused downloads to remove right now.'),
    ).toBeVisible();
  });
  it('lets the user keep the downloads after the first click', async () => {
    const fetch = stub_backend(dataset_storage);
    const user = render_card();
    await user.click(
      await screen.findByRole('button', { name: 'Remove unused downloads' }),
    );
    await user.click(screen.getByRole('button', { name: 'Keep downloads' }));
    expect(
      screen.getByRole('button', { name: 'Remove unused downloads' }),
    ).toBeEnabled();
    expect(
      fetch.mock.calls.filter(([, options]) => options?.method === 'DELETE'),
    ).toHaveLength(0);
  });
});
