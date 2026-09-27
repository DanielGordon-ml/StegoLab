import { RequestFailure } from './errors';
import { ResponseFailure } from './transport';
import {
  UPLOAD_CANCELLED_MESSAGE,
  complete_upload,
  create_upload_session,
  discard_upload,
  read_upload_session,
  upload_chunk,
} from './dataset_upload';
import type { DatasetUploadSession } from './datasets';

/** Browser storage key for the one archive upload that may be resumed. */
export const UPLOAD_MEMORY_KEY = 'stegolab_dataset_upload';
/** Refusals that ask for a later retry; every other refusal ends the session. */
const TRANSIENT_STATUSES = new Set([408, 429]);

/** What the browser remembers about an unfinished archive upload. */
export interface RememberedUpload {
  identifier: string;
  file_name: string;
  size: number;
  last_modified: number;
}

export interface UploadFlowOptions {
  /** Share of the whole archive that has reached the backend, from 0 to 1. */
  on_progress?: (fraction: number) => void;
  /** Called once the session is known, so a resume notice can show its parts. */
  on_session?: (session: DatasetUploadSession, resumed: boolean) => void;
  signal?: AbortSignal;
}

/** Read the remembered upload, if the browser still has one. */
export function read_remembered_upload(): RememberedUpload | null {
  try {
    const raw = sessionStorage.getItem(UPLOAD_MEMORY_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    return is_remembered(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/** Drop the remembered upload; later uploads of the file start again. */
export function forget_remembered_upload() {
  try {
    sessionStorage.removeItem(UPLOAD_MEMORY_KEY);
  } catch {
    // Browser storage is unavailable; there is nothing to forget.
  }
}

/** True when the remembered upload describes exactly this file. */
export function matches_file(
  memory: RememberedUpload | null,
  file: File,
): memory is RememberedUpload {
  return (
    memory !== null &&
    memory.file_name === file.name &&
    memory.size === file.size &&
    memory.last_modified === file.lastModified
  );
}

/** Keep the file name inside the backend rule without losing its extension. */
export function safe_file_name(name: string) {
  const cleaned = name
    .replace(/[^A-Za-z0-9 ._-]/g, '_')
    .replace(/^[^A-Za-z0-9]+/, '')
    .slice(0, 255);
  return cleaned || 'archive.zip';
}

/** Send one archive in 16 MiB parts, resuming an unfinished upload of the same file. */
export async function upload_dataset_archive(
  file: File,
  options: UploadFlowOptions = {},
): Promise<DatasetUploadSession> {
  const { on_progress, signal } = options;
  if (file.size < 1)
    throw new RequestFailure(
      'The archive file is empty. Choose a zip or tar archive that contains images.',
    );
  throw_if_aborted(signal);
  const { session, resumed } = await resume_or_create(file, options);
  options.on_session?.(session, resumed);
  const received = new Set(session.received_chunks);
  const part_size = (index: number) =>
    Math.min(session.chunk_bytes, file.size - index * session.chunk_bytes);
  const received_bytes = () =>
    [...received].reduce((total, index) => total + part_size(index), 0);
  const report = (in_flight: number) =>
    on_progress?.(Math.min(1, (received_bytes() + in_flight) / file.size));
  try {
    report(0);
    for (let index = 0; index < session.chunk_count; index += 1) {
      if (received.has(index)) continue;
      throw_if_aborted(signal);
      const start = index * session.chunk_bytes;
      const part = file.slice(start, start + part_size(index));
      await send_part(session.upload_identifier, index, part, report, signal);
      received.add(index);
      report(0);
    }
    throw_if_aborted(signal);
    const completed = await complete_upload(session.upload_identifier);
    // A cancel that arrived while the backend assembled the archive still counts.
    throw_if_aborted(signal);
    forget_remembered_upload();
    return completed;
  } catch (error) {
    if (signal?.aborted) {
      await abandon_upload(session.upload_identifier);
      throw new RequestFailure(UPLOAD_CANCELLED_MESSAGE);
    }
    if (error instanceof ResponseFailure && error.status === 404)
      forget_remembered_upload();
    else if (ends_session(error))
      await abandon_upload(session.upload_identifier);
    throw error;
  }
}

/** True for a refusal the backend would repeat, so this session can never finish. */
function ends_session(error: unknown) {
  return (
    error instanceof ResponseFailure &&
    error.status >= 400 &&
    error.status < 500 &&
    !TRANSIENT_STATUSES.has(error.status)
  );
}

/** Remove the session from the backend where possible and from browser memory. */
async function abandon_upload(identifier: string) {
  await discard_upload(identifier).catch(() => undefined);
  forget_remembered_upload();
}

/** Reuse the remembered session for this file when the backend still holds it. */
async function resume_or_create(file: File, options: UploadFlowOptions) {
  const memory = read_remembered_upload();
  if (matches_file(memory, file)) {
    try {
      const session = await read_upload_session(memory.identifier);
      if (!session.complete) return { session, resumed: true };
    } catch (error) {
      if (!(error instanceof ResponseFailure && error.status === 404))
        throw error;
    }
    forget_remembered_upload();
  }
  throw_if_aborted(options.signal);
  const session = await create_upload_session(
    safe_file_name(file.name),
    file.size,
  );
  remember_upload({
    identifier: session.upload_identifier,
    file_name: file.name,
    size: file.size,
    last_modified: file.lastModified,
  });
  return { session, resumed: false };
}

/** Send one part and make sure the backend stored exactly the bytes that left the browser. */
async function send_part(
  identifier: string,
  index: number,
  part: Blob,
  report: (in_flight: number) => void,
  signal?: AbortSignal,
) {
  const expected = await digest_of(part);
  const chunk = await upload_chunk(
    identifier,
    index,
    part,
    (loaded) => report(loaded),
    signal,
  );
  if (expected !== null && chunk.sha256 !== expected) {
    // The backend keeps the damaged bytes and refuses a second copy of the same
    // part, so only a fresh upload can repair it.
    await abandon_upload(identifier);
    throw new RequestFailure(
      `Part ${index + 1} of the archive arrived damaged. Check the connection, then choose the archive again to start a new upload.`,
    );
  }
  return chunk;
}

/** Hex digest of one part, or null when the browser cannot compute digests. */
async function digest_of(part: Blob): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  try {
    const buffer = await subtle.digest('SHA-256', await bytes_of(part));
    return Array.from(new Uint8Array(buffer), (byte) =>
      byte.toString(16).padStart(2, '0'),
    ).join('');
  } catch {
    return null;
  }
}

/** Read one part into memory; older browsers only offer FileReader for this. */
function bytes_of(part: Blob): Promise<ArrayBuffer> {
  if (typeof part.arrayBuffer === 'function') return part.arrayBuffer();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(part);
  });
}

function throw_if_aborted(signal?: AbortSignal) {
  if (signal?.aborted) throw new RequestFailure(UPLOAD_CANCELLED_MESSAGE);
}

function is_remembered(value: unknown): value is RememberedUpload {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as Record<string, unknown>;
  return (
    typeof record.identifier === 'string' &&
    typeof record.file_name === 'string' &&
    typeof record.size === 'number' &&
    typeof record.last_modified === 'number'
  );
}

function remember_upload(memory: RememberedUpload) {
  try {
    sessionStorage.setItem(UPLOAD_MEMORY_KEY, JSON.stringify(memory));
  } catch {
    // Browser storage is unavailable; the upload simply cannot be resumed later.
  }
}
