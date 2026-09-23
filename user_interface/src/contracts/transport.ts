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
  if (!response.ok) {
    if (validate_error(payload)) {
      throw new RequestFailure(
        payload.error.message,
        response.status >= 500,
        payload.error.diagnostic_reference,
      );
    }
    throw new RequestFailure(
      'The backend could not complete this request. Try again.',
      true,
    );
  }
  if (!validate(payload)) {
    throw new RequestFailure(
      'The backend sent an unexpected response. Check the application versions.',
      true,
    );
  }
  return payload;
}
