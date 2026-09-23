import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { expect, it, vi } from 'vitest';
import { ConfigurationPanel } from '../components/configuration_panel';

/** A late read must not overwrite the saved result in the shared cache. */
it('cancels reads racing with a successful configuration write', async () => {
  const original = { schema_version: 1, checkpoint_interval_seconds: 300 };
  const saved = { schema_version: 1, checkpoint_interval_seconds: 600 };
  let finish_read: (response: Response) => void = () => {};
  let finish_write: (response: Response) => void = () => {};
  let read_count = 0;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, options?: RequestInit) => {
      if (options?.method === 'PUT')
        return new Promise<Response>((resolve) => {
          finish_write = resolve;
        });
      read_count += 1;
      if (read_count === 1) return Response.json(original);
      return new Promise<Response>((resolve) => {
        finish_read = resolve;
      });
    }),
  );
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <ConfigurationPanel />
    </QueryClientProvider>,
  );
  const user = userEvent.setup();
  const input = await screen.findByRole('spinbutton');
  await user.clear(input);
  await user.type(input, '10');
  await user.click(screen.getByRole('button', { name: /Save settings/ }));
  void client.refetchQueries({ queryKey: ['configuration'] });
  await act(async () => {
    finish_write(Response.json(saved));
  });
  expect(await screen.findByText('Settings saved.')).toBeVisible();
  await act(async () => {
    finish_read(Response.json(original));
  });
  expect(client.getQueryData(['configuration'])).toEqual(saved);
  expect(screen.getByRole('button', { name: /Save settings/ })).toBeDisabled();
});
