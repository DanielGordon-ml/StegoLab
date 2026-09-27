import session_schema from '../../../contracts/entities/DatasetUploadSession.json';
import chunk_schema from '../../../contracts/entities/DatasetUploadChunk.json';
import { create_validator } from './validation';
import {
  UNEXPECTED_MESSAGE,
  UNREACHABLE_MESSAGE,
  ResponseFailure,
  failure_for_status,
  request_no_content,
  request_validated,
} from './transport';
import { RequestFailure } from './errors';
import type { DatasetUploadChunk, DatasetUploadSession } from './datasets';

/** One 16 MiB part may travel over a slow link, so it waits longer than a read. */
export const CHUNK_TIMEOUT_MILLISECONDS = 120000;
/** Completing may check the whole archive on the backend computer. */
export const COMPLETE_TIMEOUT_MILLISECONDS = 60000;
export const UPLOAD_CANCELLED_MESSAGE = 'The upload was cancelled.';
export const PART_TOO_LARGE_MESSAGE =
  'The server refused an upload part as too large. Choose a smaller archive or raise the server upload limit.';

export const validate_upload_session =
  create_validator<DatasetUploadSession>(session_schema);
export const validate_upload_chunk =
  create_validator<DatasetUploadChunk>(chunk_schema);

/** Path of one upload session; identifiers are always encoded. */
function session_path(identifier: string) {
  return `/datasets/uploads/${encodeURIComponent(identifier)}`;
}

/** Open a new upload session for one archive file. */
export function create_upload_session(
  file_name: string,
  total_bytes: number,
  expected_sha256?: string | null,
) {
  return request_validated('/datasets/uploads', validate_upload_session, {
    method: 'POST',
    body: JSON.stringify({
      client_request_identifier: crypto.randomUUID(),
      file_name,
      total_bytes,
      expected_sha256: expected_sha256 ?? null,
    }),
  });
}

/** Read which parts the backend already holds for one session. */
export function read_upload_session(identifier: string) {
  return request_validated(session_path(identifier), validate_upload_session);
}

/** Tell the backend that every part was sent so it can assemble the archive. */
export function complete_upload(identifier: string) {
  return request_validated(
    `${session_path(identifier)}/complete`,
    validate_upload_session,
    {
      method: 'POST',
      body: JSON.stringify({ client_request_identifier: crypto.randomUUID() }),
    },
    COMPLETE_TIMEOUT_MILLISECONDS,
  );
}

/** Remove an unfinished or unwanted upload from the backend computer. */
export function discard_upload(identifier: string) {
  return request_no_content(session_path(identifier), { method: 'DELETE' });
}

/** Send one archive part as the raw request body, reporting progress and safe failures. */
export function upload_chunk(
  identifier: string,
  index: number,
  part: Blob,
  on_progress?: (loaded_bytes: number, total_bytes: number) => void,
  signal?: AbortSignal,
): Promise<DatasetUploadChunk> {
  if (signal?.aborted)
    return Promise.reject(new RequestFailure(UPLOAD_CANCELLED_MESSAGE));
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('PUT', `/api/v1${session_path(identifier)}/chunks/${index}`);
    request.timeout = CHUNK_TIMEOUT_MILLISECONDS;
    request.responseType = 'text';
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    const stop = () => request.abort();
    signal?.addEventListener('abort', stop, { once: true });
    /** Settle once and stop listening for a cancellation that can no longer apply. */
    const settle = (outcome: () => void) => {
      signal?.removeEventListener('abort', stop);
      outcome();
    };
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) on_progress?.(event.loaded, event.total);
    };
    request.onerror = () =>
      settle(() => reject(new RequestFailure(UNREACHABLE_MESSAGE, true)));
    request.ontimeout = () =>
      settle(() =>
        reject(
          new RequestFailure(
            'Sending this part took too long. Check the connection and try again.',
            true,
          ),
        ),
      );
    request.onabort = () =>
      settle(() => reject(new RequestFailure(UPLOAD_CANCELLED_MESSAGE)));
    request.onload = () => settle(() => answer(request, resolve, reject));
    request.send(part);
  });
}

/** Map the finished part request to a validated record or a safe failure. */
function answer(
  request: XMLHttpRequest,
  resolve: (chunk: DatasetUploadChunk) => void,
  reject: (failure: RequestFailure) => void,
) {
  if (request.status === 413) {
    // A proxy answered without the usual error body; keep the status for callers.
    reject(new ResponseFailure(413, PART_TOO_LARGE_MESSAGE));
    return;
  }
  let payload: unknown = null;
  try {
    payload = JSON.parse(request.responseText);
  } catch {
    payload = null;
  }
  if (request.status >= 200 && request.status < 300) {
    if (validate_upload_chunk(payload)) resolve(payload);
    else reject(new RequestFailure(UNEXPECTED_MESSAGE, true));
    return;
  }
  reject(failure_for_status(request.status, payload));
}
