import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { Application } from '../application';
import { parse_minutes } from '../components/configuration_panel';
import { mock_backend, saved_configuration } from './fixtures';

/** Mount the real shell with a fresh query cache. */
function render_application() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <Application />
    </QueryClientProvider>,
  );
  return userEvent.setup();
}

/** Open Config after the backend response has arrived. */
async function open_configuration(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('tab', { name: /Config/ }));
  return screen.findByRole('spinbutton', { name: 'Checkpoint frequency' });
}

describe('workspace', () => {
  it('shows truthful unavailable states and supports arrow, Home and End navigation', async () => {
    mock_backend();
    const user = render_application();
    expect(await screen.findByText('Backend connected')).toBeVisible();
    expect(
      screen.getByText('Image encoding is not available yet.'),
    ).toBeVisible();
    await user.click(screen.getByRole('tab', { name: 'Encode' }));
    await user.keyboard('{ArrowRight}');
    expect(screen.getByRole('tab', { name: 'Decode' })).toHaveFocus();
    expect(
      screen.getByText('Image decoding is not available yet.'),
    ).toBeVisible();
    await user.keyboard('{End}');
    expect(screen.getByRole('tab', { name: /Config/ })).toHaveFocus();
    await user.keyboard('{Home}');
    expect(screen.getByRole('tab', { name: 'Encode' })).toHaveFocus();
  });
  it('confirms only successful saves and keeps edits when switching tabs', async () => {
    let complete_save: (response: Response) => void = () => {};
    const fetch = mock_backend(
      () =>
        new Promise((resolve) => {
          complete_save = resolve;
        }),
    );
    const user = render_application();
    const input = await open_configuration(user);
    await user.clear(input);
    await user.type(input, '7');
    await user.click(screen.getByRole('tab', { name: 'Train' }));
    await user.click(screen.getByRole('tab', { name: /Config/ }));
    expect(input).toHaveValue(7);
    await user.click(screen.getByRole('button', { name: /Save settings/ }));
    expect(screen.queryByText('Settings saved.')).not.toBeInTheDocument();
    expect(screen.getByText('Saving settings…')).toBeVisible();
    complete_save(
      Response.json({
        ...saved_configuration,
        checkpoint_interval_seconds: 420,
      }),
    );
    expect(await screen.findByText('Settings saved.')).toBeVisible();
    const writes = fetch.mock.calls.filter(
      ([, options]) => options?.method === 'PUT',
    );
    expect(
      JSON.parse(String(writes[0][1]?.body)).configuration
        .checkpoint_interval_seconds,
    ).toBe(420);
  });
  it('retries an uncertain save with the same identifier and submitted value', async () => {
    let attempts = 0;
    const fetch = mock_backend(async () => {
      attempts += 1;
      if (attempts === 1) throw new TypeError('network-private-test-value');
      return Response.json({
        ...saved_configuration,
        checkpoint_interval_seconds: 540,
      });
    });
    const user = render_application();
    const input = await open_configuration(user);
    await user.clear(input);
    await user.type(input, '9');
    await user.click(screen.getByRole('button', { name: /Save settings/ }));
    const retry = await screen.findByRole('button', { name: 'Retry save' });
    expect(input).toHaveValue(9);
    expect(input).toBeDisabled();
    expect(
      screen.queryByText(/network-private-test-value/),
    ).not.toBeInTheDocument();
    await user.click(retry);
    expect(await screen.findByText('Settings saved.')).toBeVisible();
    const writes = fetch.mock.calls.filter(
      ([, options]) => options?.method === 'PUT',
    );
    expect(writes[0][1]?.body).toBe(writes[1][1]?.body);
  });
  it('saves reset defaults and reports initial connection failures', async () => {
    mock_backend();
    const user = render_application();
    const input = await open_configuration(user);
    await user.clear(input);
    await user.type(input, '11');
    await user.click(screen.getByRole('button', { name: 'Reset to defaults' }));
    expect(
      await screen.findByText('Defaults restored and saved.'),
    ).toBeVisible();
    expect(input).toHaveValue(5);
  });
  it('shows a retry action for an unavailable backend without invented settings', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('offline');
      }),
    );
    const user = render_application();
    await user.click(screen.getByRole('tab', { name: /Config/ }));
    await waitFor(() => expect(screen.getByText('Disconnected')).toBeVisible());
    const panel = screen.getByRole('tabpanel', { name: /Config/ });
    expect(
      within(panel).getByRole('button', { name: 'Retry loading settings' }),
    ).toBeVisible();
    expect(within(panel).queryByRole('spinbutton')).not.toBeInTheDocument();
  });
});

it('accepts only safe positive whole minutes', () => {
  expect(parse_minutes('5')).toBe(300);
  for (const invalid of ['', '0', '-1', '1.5', '1e2', ' 2', '9007199254740991'])
    expect(parse_minutes(invalid)).toBeNull();
});
