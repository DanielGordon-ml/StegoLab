import type { ValidateFunction } from './validation';
import { RequestFailure } from './errors';
import { validate_error } from './validation';

/** Fetch and validate one response without exposing its raw content in errors. */
export async function request_validated<Result>(
  path: string,
  validate: ValidateFunction<Result>,
  options: RequestInit = {},
): Promise<Result> {
  let response: Response;
  let payload: unknown;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...options,
      cache: 'no-store',
      signal: AbortSignal.timeout(10000),
      headers: { 'Content-Type': 'application/json', ...options.headers },
    });
    payload = await response.json();
  } catch {
    throw new RequestFailure(
      'Cannot reach StegoLab. Check that the backend is running, then try again.',
      true,
    );
  }
  if (!response.ok) throw failure_for(response, payload);
  if (!validate(payload)) {
    throw new RequestFailure(
      'The backend sent an unexpected response. Check the application versions.',
      true,
    );
  }
  return payload;
}

/** Turn a failed response into a safe failure without echoing its body. */
export function failure_for(response: Response, payload: unknown) {
  if (validate_error(payload)) {
    return new RequestFailure(
      payload.error.message,
      response.status >= 500,
      payload.error.diagnostic_reference,
    );
  }
  return new RequestFailure(
    'The backend could not complete this request. Try again.',
    true,
  );
}

/** Send a request whose success answer has no body, such as a delete. */
export async function request_no_content(
  path: string,
  options: RequestInit = {},
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...options,
      cache: 'no-store',
      signal: AbortSignal.timeout(10000),
    });
  } catch {
    throw new RequestFailure(
      'Cannot reach StegoLab. Check that the backend is running, then try again.',
      true,
    );
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
