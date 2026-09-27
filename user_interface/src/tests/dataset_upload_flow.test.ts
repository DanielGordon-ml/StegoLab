import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FakeRequest } from './fake_request';
import {
  UPLOAD_MEMORY_KEY,
  upload_dataset_archive,
} from '../contracts/dataset_upload_flow';

const CHUNK = 16777216;
const SIZE = CHUNK + 1048576;
const LAST_MODIFIED = 1700000000000;
const identifier = `upload_${'0'.repeat(31)}1`;
const file = new File([new Uint8Array(SIZE)], 'photos.zip', {
  type: 'application/zip',
  lastModified: LAST_MODIFIED,
});
const memory = {
  identifier,
  file_name: 'photos.zip',
  size: SIZE,
  last_modified: LAST_MODIFIED,
};

/** Session record with every field the response contract requires. */
function session(received: number[], complete = false) {
  return {
    upload_identifier: identifier,
    file_name: 'photos.zip',
    total_bytes: SIZE,
    chunk_bytes: CHUNK,
    chunk_count: 2,
    received_chunks: received,
    complete,
    sha256: null,
    created_at: '2026-09-27T10:00:00Z',
    expires_at: '2026-09-28T10:00:00Z',
  };
}
/** Stand-in digest: the byte length written into the last bytes of 32. */
function fake_digest(data: ArrayBuffer) {
  const bytes = new Uint8Array(32);
  let remaining = data.byteLength;
  for (let position = 31; position >= 0 && remaining > 0; position -= 1) {
    bytes[position] = remaining % 256;
    remaining = Math.floor(remaining / 256);
  }
  return bytes.buffer;
}
const fake_hex = (length: number) => length.toString(16).padStart(64, '0');
/** Error body the backend sends, with the code and message under test. */
const refusal = (status: number, code: string, message: string) =>
  Response.json(
    {
      error: {
        code,
        message,
        diagnostic_reference: '0123456789abcdef0123456789abcdef',
      },
    },
    { status },
  );
const not_found = () =>
  refusal(404, 'upload_expired', 'This upload expired. Start it again.');
const completed = () => Response.json(session([0, 1], true));

const backend = { read: not_found, complete: completed };
const calls: { method: string; url: string; body: unknown }[] = [];
const digest = vi.fn(async (_algorithm: string, data: ArrayBuffer) =>
  fake_digest(data),
);

beforeEach(() => {
  backend.read = not_found;
  backend.complete = completed;
  calls.length = 0;
  sessionStorage.clear();
  FakeRequest.reset();
  vi.stubGlobal('XMLHttpRequest', FakeRequest);
  vi.stubGlobal('crypto', {
    randomUUID: () => 'request-1',
    subtle: { digest },
  });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, options?: RequestInit) => {
      const method = options?.method ?? 'GET';
      const body = options?.body ? JSON.parse(String(options.body)) : null;
      calls.push({ method, url, body });
      if (method === 'GET') return backend.read();
      if (method === 'DELETE') return new Response(null, { status: 204 });
      if (url.endsWith('/complete')) return backend.complete();
      return Response.json(session([]), { status: 201 });
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

/** Answer one part the way the backend would, with a matching digest by default. */
async function answer_part(index: number, received: number[], sha256?: string) {
  const request = await FakeRequest.next_send();
  const bytes = (request.sent as Blob).size;
  request.respond(200, {
    upload_identifier: identifier,
    index,
    bytes,
    sha256: sha256 ?? fake_hex(bytes),
    received_chunks: received,
  });
  return request;
}
const traffic = () => calls.map((call) => `${call.method} ${call.url}`);
const discard = `DELETE /api/v1/datasets/uploads/${identifier}`;
const remembered = () => sessionStorage.getItem(UPLOAD_MEMORY_KEY);

describe('dataset archive upload flow', () => {
  it('sends both parts in order as raw bytes and reports growing progress', async () => {
    const progress: number[] = [];
    const flow = upload_dataset_archive(file, {
      on_progress: (fraction) => progress.push(fraction),
    });
    const first = await FakeRequest.next_send();
    expect(first.method).toBe('PUT');
    expect(first.url).toBe(`/api/v1/datasets/uploads/${identifier}/chunks/0`);
    expect(first.headers['Content-Type']).toBe('application/octet-stream');
    expect((first.sent as Blob).size).toBe(CHUNK);
    first.upload.onprogress?.({
      lengthComputable: true,
      loaded: CHUNK / 2,
      total: CHUNK,
    } as ProgressEvent);
    first.respond(200, {
      upload_identifier: identifier,
      index: 0,
      bytes: CHUNK,
      sha256: fake_hex(CHUNK),
      received_chunks: [0],
    });
    const second = await answer_part(1, [0, 1]);
    expect(second.url).toBe(`/api/v1/datasets/uploads/${identifier}/chunks/1`);
    expect((second.sent as Blob).size).toBe(SIZE - CHUNK);
    await expect(flow).resolves.toMatchObject({ complete: true });
    expect(progress[0]).toBe(0);
    expect(progress.at(-1)).toBe(1);
    expect(
      progress.every((value, at) => at === 0 || value >= progress[at - 1]),
    ).toBe(true);
    expect(traffic()).toEqual([
      'POST /api/v1/datasets/uploads',
      `POST /api/v1/datasets/uploads/${identifier}/complete`,
    ]);
    expect(calls[0].body).toMatchObject({
      file_name: 'photos.zip',
      total_bytes: SIZE,
      expected_sha256: null,
    });
    expect(remembered()).toBeNull();
  });
  it('resumes a remembered session and sends only the missing part', async () => {
    sessionStorage.setItem(UPLOAD_MEMORY_KEY, JSON.stringify(memory));
    backend.read = () => Response.json(session([0]));
    const on_session = vi.fn();
    const progress: number[] = [];
    const flow = upload_dataset_archive(file, {
      on_session,
      on_progress: (fraction) => progress.push(fraction),
    });
    const only = await answer_part(1, [0, 1]);
    expect(only.url).toBe(`/api/v1/datasets/uploads/${identifier}/chunks/1`);
    await expect(flow).resolves.toMatchObject({ complete: true });
    expect(FakeRequest.sent_requests).toHaveLength(1);
    expect(on_session).toHaveBeenCalledWith(
      expect.objectContaining({ received_chunks: [0] }),
      true,
    );
    expect(progress[0]).toBeCloseTo(CHUNK / SIZE);
    expect(traffic()).toEqual([
      `GET /api/v1/datasets/uploads/${identifier}`,
      `POST /api/v1/datasets/uploads/${identifier}/complete`,
    ]);
  });
  it('discards the session and forgets it when a part arrives damaged', async () => {
    const flow = upload_dataset_archive(file);
    await answer_part(0, [0], 'f'.repeat(64));
    await expect(flow).rejects.toThrow('Part 1 of the archive arrived damaged');
    // The backend keeps the damaged part, so the same index is never sent again.
    expect(FakeRequest.sent_requests).toHaveLength(1);
    expect(digest).toHaveBeenCalledTimes(1);
    expect(traffic()).toContain(discard);
    expect(remembered()).toBeNull();
  });
  it('forgets the session when the backend refuses a part for good', async () => {
    const flow = upload_dataset_archive(file);
    const first = await FakeRequest.next_send();
    first.respond(422, {
      error: {
        code: 'upload_chunk_size',
        message: 'This part has the wrong size. Start the upload again.',
        diagnostic_reference: '0123456789abcdef0123456789abcdef',
      },
    });
    await expect(flow).rejects.toThrow('wrong size');
    expect(traffic()).toContain(discard);
    expect(remembered()).toBeNull();
  });
  it('cancels through the signal, discards the session and forgets it', async () => {
    const controller = new AbortController();
    const flow = upload_dataset_archive(file, { signal: controller.signal });
    const first = await FakeRequest.next_send();
    controller.abort();
    expect(first.aborted).toBe(true);
    await expect(flow).rejects.toThrow('The upload was cancelled.');
    expect(traffic()).toContain(discard);
    expect(remembered()).toBeNull();
  });
  it('treats a cancel during completion as cancelled and discards the archive', async () => {
    const controller = new AbortController();
    backend.complete = () => {
      controller.abort();
      return completed();
    };
    const flow = upload_dataset_archive(file, { signal: controller.signal });
    await answer_part(0, [0]);
    await answer_part(1, [0, 1]);
    await expect(flow).rejects.toThrow('The upload was cancelled.');
    expect(traffic()).toContain(discard);
    expect(remembered()).toBeNull();
  });
  it('maps a proxy size refusal to a plain message', async () => {
    const flow = upload_dataset_archive(file);
    const first = await FakeRequest.next_send();
    first.respond(413, '<html>413 Request Entity Too Large</html>');
    await expect(flow).rejects.toThrow('too large');
    expect(remembered()).toBeNull();
  });
  it('starts a new session when the remembered one has expired', async () => {
    const stale = `upload_${'a'.repeat(32)}`;
    sessionStorage.setItem(
      UPLOAD_MEMORY_KEY,
      JSON.stringify({ ...memory, identifier: stale }),
    );
    const flow = upload_dataset_archive(file);
    const first = await FakeRequest.next_send();
    expect(first.url).toBe(`/api/v1/datasets/uploads/${identifier}/chunks/0`);
    expect(JSON.parse(remembered() ?? '{}')).toMatchObject({ identifier });
    first.respond(200, {
      upload_identifier: identifier,
      index: 0,
      bytes: CHUNK,
      sha256: fake_hex(CHUNK),
      received_chunks: [0],
    });
    await answer_part(1, [0, 1]);
    await expect(flow).resolves.toMatchObject({ complete: true });
    expect(traffic()).toEqual([
      `GET /api/v1/datasets/uploads/${stale}`,
      'POST /api/v1/datasets/uploads',
      `POST /api/v1/datasets/uploads/${identifier}/complete`,
    ]);
  });
});
