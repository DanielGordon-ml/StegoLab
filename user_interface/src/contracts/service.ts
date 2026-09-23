import type { ConfigurationReset, ConfigurationUpdate } from './configuration';
import { RequestFailure } from './errors';
import { request_validated } from './transport';
import {
  validate_capabilities,
  validate_configuration,
  validate_health,
  validate_reset,
  validate_update,
} from './validation';

/** Read the current server configuration. */
export function read_configuration() {
  return request_validated('/configuration', validate_configuration);
}

/** Read supported actions and resources. */
export function read_capabilities() {
  return request_validated('/capabilities', validate_capabilities);
}

/** Check that the local backend is ready. */
export function read_health() {
  return request_validated('/health', validate_health);
}

/** Persist a structurally validated configuration change. */
export function save_configuration(request: ConfigurationUpdate) {
  if (!validate_update(request)) {
    throw new RequestFailure('Enter a positive whole number of minutes.');
  }
  return request_validated('/configuration', validate_configuration, {
    method: 'PUT',
    body: JSON.stringify(request),
  });
}

/** Persist the factory default using a retry-safe request. */
export function reset_configuration(request: ConfigurationReset) {
  if (!validate_reset(request)) {
    throw new RequestFailure(
      'The request could not be prepared. Reload and try again.',
    );
  }
  return request_validated('/configuration/reset', validate_configuration, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}
