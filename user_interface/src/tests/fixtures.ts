import { vi } from 'vitest';
import { empty_workspace } from './workflow_fixtures';

export const saved_configuration = {
  schema_version: 1,
  checkpoint_interval_seconds: 300,
};
export const capabilities = {
  application_version: '0.1.0',
  available_devices: ['cpu'],
  available_models: [],
  available_profiles: [],
  encoding_available: false,
  decoding_available: false,
  training_available: false,
  maximum_payload_bytes: 0,
};

/** Serve real contract-shaped reads, delegating mutation behavior to each test. */
export function mock_backend(
  mutate?: (options: RequestInit) => Promise<Response>,
) {
  const fetch = vi.fn(
    async (url: string | URL | Request, options?: RequestInit) => {
      if (options?.method)
        return mutate ? mutate(options) : Response.json(saved_configuration);
      if (String(url).endsWith('/health'))
        return Response.json({ status: 'ready', application_version: '0.1.0' });
      if (String(url).endsWith('/capabilities'))
        return Response.json(capabilities);
      if (String(url).endsWith('/workspace'))
        return Response.json(empty_workspace);
      if (String(url).endsWith('/jobs')) return Response.json({ items: [] });
      return Response.json(saved_configuration);
    },
  );
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
