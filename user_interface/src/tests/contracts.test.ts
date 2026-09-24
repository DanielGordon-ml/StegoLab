import { describe, expect, it } from 'vitest';
import {
  validate_capabilities,
  validate_configuration,
  validate_error,
  validate_health,
  validate_update,
} from '../contracts/validation';
import { capabilities, saved_configuration } from './fixtures';
import { request_validated } from '../contracts/transport';
import { vi } from 'vitest';

describe('backend JSON Schemas', () => {
  it('accepts complete responses and rejects missing, coerced, and extra fields', () => {
    expect(validate_configuration(saved_configuration)).toBe(true);
    for (const invalid of [
      {},
      { ...saved_configuration, checkpoint_interval_seconds: '300' },
      { ...saved_configuration, checkpoint_interval_seconds: 1 },
      { ...saved_configuration, checkpoint_interval_seconds: 0 },
      { ...saved_configuration, checkpoint_interval_seconds: 60.5 },
      { ...saved_configuration, password: 'secret' },
    ]) {
      expect(validate_configuration(invalid)).toBe(false);
    }
    expect(validate_capabilities(capabilities)).toBe(true);
    expect(validate_capabilities({})).toBe(false);
    expect(
      validate_capabilities({ ...capabilities, training_available: 'false' }),
    ).toBe(false);
    expect(
      validate_capabilities({ ...capabilities, maximum_payload_bytes: 4096 }),
    ).toBe(false);
    expect(
      validate_health({ status: 'ready', application_version: '0.1.0' }),
    ).toBe(true);
    expect(validate_health({})).toBe(false);
  });
  it('checks nested changes and safe error envelopes against shared schemas', () => {
    const change = {
      client_request_identifier: 'example-request',
      configuration: saved_configuration,
    };
    expect(validate_update(change)).toBe(true);
    expect(
      validate_update({
        ...change,
        configuration: { ...saved_configuration, unknown: true },
      }),
    ).toBe(false);
    expect(validate_update({ ...change, client_request_identifier: '' })).toBe(
      false,
    );
    expect(
      validate_error({
        error: {
          code: 'unavailable',
          message: 'Try again.',
          diagnostic_reference: '0123456789abcdef0123456789abcdef',
        },
      }),
    ).toBe(true);
    expect(
      validate_error({ error: { code: 'unavailable', message: 'Try again.' } }),
    ).toBe(false);
  });
  it('never exposes an untrusted response body in an error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => Response.json({ secret: 'private-test-value' })),
    );
    await expect(
      request_validated('/configuration', validate_configuration),
    ).rejects.toThrow('unexpected response');
    await expect(
      request_validated('/configuration', validate_configuration),
    ).rejects.not.toThrow('private-test-value');
  });
});
