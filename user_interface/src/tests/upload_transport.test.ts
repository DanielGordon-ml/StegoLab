import { afterEach, describe, expect, it, vi } from 'vitest';
import { RequestFailure } from '../contracts/errors';
import { upload_image } from '../contracts/upload';
import { uploaded_image } from './fixtures';

/** Stand in for the browser request so each outcome can be driven by hand. */
class FakeRequest {
  static current: FakeRequest;
  status = 0;
  responseText = '';
  timeout = 0;
  responseType = '';
  headers: Record<string, string> = {};
  upload = { onprogress: null as ((event: ProgressEvent) => void) | null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  ontimeout: (() => void) | null = null;
  onabort: (() => void) | null = null;
  sent: unknown = null;
  url = '';
  constructor() {
    FakeRequest.current = this;
  }
  open(_method: string, url: string) {
    this.url = url;
  }
  setRequestHeader(name: string, value: string) {
    this.headers[name] = value;
  }
  send(body: unknown) {
    this.sent = body;
  }
}

const file = new File(['png'], 'cover.png', { type: 'image/png' });
afterEach(() => vi.unstubAllGlobals());

/** Start an upload against the fake request and hand back both. */
function start(on_progress?: (fraction: number) => void) {
  vi.stubGlobal('XMLHttpRequest', FakeRequest);
  const promise = upload_image(file, 'cover', on_progress);
  return { promise, request: FakeRequest.current };
}

describe('upload transport', () => {
  it('reports progress and resolves a validated upload record', async () => {
    const progress = vi.fn();
    const { promise, request } = start(progress);
    expect(request.url).toBe('/api/v1/images?purpose=cover');
    expect(request.headers['Content-Type']).toBe('image/png');
    expect(request.sent).toBe(file);
    request.upload.onprogress?.({
      lengthComputable: true,
      loaded: 5,
      total: 10,
    } as ProgressEvent);
    expect(progress).toHaveBeenCalledWith(0.5);
    request.status = 201;
    request.responseText = JSON.stringify(uploaded_image);
    request.onload?.();
    await expect(promise).resolves.toEqual(uploaded_image);
  });
  it('maps the proxy size limit, error envelopes, and timeouts to safe messages', async () => {
    const too_large = start();
    too_large.request.status = 413;
    too_large.request.responseText =
      '<html>413 Request Entity Too Large</html>';
    too_large.request.onload?.();
    await expect(too_large.promise).rejects.toThrow('16 MiB upload limit');
    const refused = start();
    refused.request.status = 422;
    refused.request.responseText = JSON.stringify({
      error: {
        code: 'image_limits',
        message: 'The image exceeds supported limits.',
        diagnostic_reference: '0123456789abcdef0123456789abcdef',
      },
    });
    refused.request.onload?.();
    await expect(refused.promise).rejects.toThrow(
      'The image exceeds supported limits.',
    );
    const timed_out = start();
    timed_out.request.ontimeout?.();
    await expect(timed_out.promise).rejects.toMatchObject({ uncertain: true });
    const unexpected = start();
    unexpected.request.status = 201;
    unexpected.request.responseText = '{"secret":"private-test-value"}';
    unexpected.request.onload?.();
    await expect(unexpected.promise).rejects.toBeInstanceOf(RequestFailure);
    await expect(unexpected.promise).rejects.not.toThrow('private-test-value');
  });
  it('refuses files above the limit before sending anything', async () => {
    const huge = new File([new Uint8Array(16777217)], 'huge.png');
    const constructed = vi.fn();
    vi.stubGlobal(
      'XMLHttpRequest',
      class {
        constructor() {
          constructed();
        }
      },
    );
    await expect(upload_image(huge, 'cover')).rejects.toThrow('16 MiB');
    expect(constructed).not.toHaveBeenCalled();
  });
});
