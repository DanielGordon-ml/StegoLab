import type { ValidateFunction } from './validation';
import { RequestFailure } from './errors';
import { validate_error } from './validation';

export const DEFAULT_TIMEOUT_MILLISECONDS = 10000;
export const UNREACHABLE_MESSAGE =
  'Cannot reach StegoLab. Check that the backend is running, then try again.';
export const UNEXPECTED_MESSAGE =
  'The backend sent an unexpected response. Check the application versions.';

/** Safe failure that also remembers the response status for callers that branch on it. */
export class ResponseFailure extends RequestFailure {
  constructor(
    readonly status: number,
    message: string,
    uncertain = false,
    diagnostic_reference?: string,
  ) {
    super(message, uncertain, diagnostic_reference);
  }
}

/** Fetch and validate one response without exposing its raw content in errors. */
export async function request_validated<Result>(
  path: string,
  validate: ValidateFunction<Result>,
  options: RequestInit = {},
  timeout_milliseconds = DEFAULT_TIMEOUT_MILLISECONDS,
): Promise<Result> {
  let response: Response;
  let payload: unknown;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...options,
      cache: 'no-store',
      signal: AbortSignal.timeout(timeout_milliseconds),
      headers: { 'Content-Type': 'application/json', ...options.headers },
    });
    payload = await response.json();
  } catch {
    throw new RequestFailure(UNREACHABLE_MESSAGE, true);
  }
  if (!response.ok) throw failure_for(response, payload);
  if (!validate(payload)) throw new RequestFailure(UNEXPECTED_MESSAGE, true);
  return payload;
}

/** Turn a failed response into a safe failure without echoing its body. */
export function failure_for(response: Response, payload: unknown) {
  return failure_for_status(response.status, payload);
}

/** Map a failed status and its parsed body to a safe failure that keeps the status. */
export function failure_for_status(status: number, payload: unknown) {
  if (validate_error(payload)) {
    return new ResponseFailure(
      status,
      payload.error.message,
      status >= 500,
      payload.error.diagnostic_reference,
    );
  }
  return new ResponseFailure(
    status,
    'The backend could not complete this request. Try again.',
    true,
  );
}

/** Send a request whose success answer has no body, such as a delete. */
export async function request_no_content(
  path: string,
  options: RequestInit = {},
  timeout_milliseconds = DEFAULT_TIMEOUT_MILLISECONDS,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...options,
      cache: 'no-store',
      signal: AbortSignal.timeout(timeout_milliseconds),
    });
  } catch {
    throw new RequestFailure(UNREACHABLE_MESSAGE, true);
  }
  if (response.ok) return;
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  throw failure_for(response, payload);
}
