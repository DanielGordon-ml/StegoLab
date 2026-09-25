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
  experimental_models_only: true,
  maximum_payload_bytes: 0,
  minimum_image_side: 0,
  maximum_image_side: 0,
  maximum_upload_bytes: 16777216,
};

export const uploaded_image = {
  image_reference: 'image_0123456789abcdef0123456789abcdef',
  purpose: 'cover',
  summary: {
    source_format: 'JPEG',
    mode: 'RGB',
    pixel_policy: 'prepare_srgb',
    source_width: 1024,
    source_height: 768,
    prepared_width: 1024,
    prepared_height: 768,
    input_size_bytes: 204800,
    orientation: 1,
    color_policy: 'assumed_srgb',
  },
  warnings: [
    'The JPEG file was converted to PNG. Encode and Decode always use PNG.',
  ],
  created_at: '2026-09-25T10:00:00+00:00',
  expires_at: '2026-09-26T10:00:00+00:00',
};

export const capacity_result = {
  image_reference: uploaded_image.image_reference,
  model_identifier: '001V_2026-09-24',
  profile_identifier: 'test_only_v1',
  width: 1024,
  height: 768,
  maximum_message_bytes: 768,
  capacity: {
    profile_identifier: 'test_only_v1',
    maximum_message_bytes: 768,
    header_bytes: 45,
    message_length_bytes: 2,
    authentication_bytes: 16,
    padding_bytes_at_capacity: 61,
    block_count: 4,
    frame_bytes: 892,
    correction_bytes: 128,
    protected_bits: 8160,
    payload_map_bits: 786432,
    repeated_bits: 778272,
    minimum_repetitions: 96,
    additional_repetitions: 3072,
  },
};

export const encoding_request = {
  client_request_identifier: 'encode-1',
  image_reference: uploaded_image.image_reference,
  model_identifier: '001V_2026-09-24',
  message: 'hello',
  password: 'example-password',
};

export const encoding_result = {
  artifact_identifier: 'encoded_0123456789abcdef0123456789abcdef',
  filename: 'stegolab-encoded-01234567.png',
  width: 1024,
  height: 768,
  png_bytes: 2359296,
  message_byte_count: 5,
  verified: true,
};

export const decoded_text = {
  job_identifier: 'job_0123456789abcdef',
  text: 'hello',
  byte_count: 5,
  expires_at: '2026-09-25T10:05:00+00:00',
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
