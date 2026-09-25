import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Application } from '../application';
import { capacity_result, uploaded_image } from './fixtures';
import { installed_model, mock_inference_backend } from './inference_fixtures';

vi.mock('../contracts/upload', () => ({
  upload_image: vi.fn(async () => uploaded_image),
}));

/** Mount the shell over a fresh cache with object URLs stubbed for jsdom. */
function render_encode() {
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
  return userEvent.setup();
}

const cover = new File(['png-bytes'], 'cover.png', { type: 'image/png' });
afterEach(() => vi.unstubAllGlobals());

describe('encode capacity', () => {
  it('counts UTF-8 bytes against the image limit and blocks oversized messages', async () => {
    mock_inference_backend();
    const user = render_encode();
    await user.click(screen.getByRole('tab', { name: 'Encode' }));
    expect(await screen.findByText('Experimental model.')).toBeVisible();
    await user.upload(screen.getByLabelText('Cover image'), cover);
    expect(await screen.findByText(/1024×768 px/)).toBeVisible();
    expect(screen.getByLabelText('Encoder model')).toHaveValue(
      installed_model.model_identifier,
    );
    const message = screen.getByLabelText('Message');
    await user.type(message, 'שלום 🙂');
    expect(await screen.findByText('13 of 768 bytes used')).toBeVisible();
    await user.type(screen.getByLabelText('Password'), 'pw');
    const encode = screen.getByRole('button', { name: 'Encode and verify' });
    expect(encode).toBeEnabled();
    await user.type(message, 'é'.repeat(380));
    expect(screen.getByText('1153 of 768 bytes used')).toHaveClass(
      'invalid_text',
    );
    expect(encode).toBeDisabled();
  });
  it('ignores a late capacity answer for a model that is no longer selected', async () => {
    const second = { ...installed_model, model_identifier: 'second_model' };
    let finish_first: (response: Response) => void = () => {};
    mock_inference_backend({
      models: [installed_model, second],
      capacity: (body) =>
        body.model_identifier === 'second_model'
          ? Response.json({
              ...capacity_result,
              model_identifier: 'second_model',
              maximum_message_bytes: 512,
              capacity: {
                ...capacity_result.capacity,
                maximum_message_bytes: 512,
              },
            })
          : new Promise<Response>((resolve) => {
              finish_first = resolve;
            }),
    });
    const user = render_encode();
    await user.click(screen.getByRole('tab', { name: 'Encode' }));
    await user.upload(await screen.findByLabelText('Cover image'), cover);
    await screen.findByText(/1024×768 px/);
    await user.selectOptions(
      screen.getByLabelText('Encoder model'),
      'second_model',
    );
    expect(await screen.findByText('0 of 512 bytes used')).toBeVisible();
    finish_first(Response.json(capacity_result));
    await waitFor(() =>
      expect(screen.getByText('0 of 512 bytes used')).toBeVisible(),
    );
    expect(screen.queryByText(/of 768 bytes/)).not.toBeInTheDocument();
  });
});
