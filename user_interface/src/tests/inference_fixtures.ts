import { vi } from 'vitest';
import type { JobSnapshot } from '../contracts/workflows';
import type { InstalledModel } from '../contracts/models';
import { capacity_result, decoded_text, uploaded_image } from './fixtures';
import { empty_workspace, saved_job } from './workflow_fixtures';

export const installed_model: InstalledModel = {
  model_identifier: '001V_2026-09-24',
  compatibility_identifier: `dense_v1_${'a'.repeat(64)}`,
  source_identifier: 'example_source',
  profile_identifier: 'test_only_v1',
  status: 'experimental',
  format_version: 2,
  minimum_side: 512,
  maximum_side: 1024,
  maximum_payload_bytes: 1024,
  installed_at: '2026-09-25T09:00:00+00:00',
  export_reference: 'export_example',
};

export const capabilities_with_model = {
  application_version: '0.1.0',
  available_devices: ['cpu'],
  available_models: [installed_model.model_identifier],
  available_profiles: ['test_only_v1'],
  encoding_available: true,
  decoding_available: true,
  training_available: true,
  experimental_models_only: true,
  maximum_payload_bytes: 1024,
  minimum_image_side: 512,
  maximum_image_side: 1024,
  maximum_upload_bytes: 16777216,
  maximum_dataset_upload_bytes: 2147483648,
  dataset_upload_chunk_bytes: 16777216,
  hugging_face_token_configured: false,
  dataset_source_kinds: ['server_folder'],
};

export const encoded_upload = {
  ...uploaded_image,
  image_reference: 'image_fedcba9876543210fedcba9876543210',
  purpose: 'encoded' as const,
  summary: {
    ...uploaded_image.summary,
    pixel_policy: 'preserve_stored' as const,
  },
  warnings: [],
};

/** A decode job as the backend reports it while queued. */
export const decode_job: JobSnapshot = {
  ...saved_job,
  job_identifier: 'job_decode',
  operation: 'decode',
  experiment_identifier: null,
  phase: 'queued',
  frozen_settings: {
    operation: 'decode',
    image_reference: encoded_upload.image_reference,
    model_identifier: installed_model.model_identifier,
    message_byte_count: 0,
    width: 1024,
    height: 768,
  },
};

export const completed_decode_job: JobSnapshot = {
  ...decode_job,
  status: 'completed',
  phase: 'completed',
  available_actions: [],
  updated_at: '2026-09-24T10:01:00Z',
  latest_event_identifier: 3,
  result: {
    result_available: true,
    text_byte_count: 5,
    expires_at: decoded_text.expires_at,
  },
};

interface BackendOptions {
  models?: InstalledModel[];
  capacity?: (body: {
    model_identifier: string;
  }) => Promise<Response> | Response;
  jobs?: () => JobSnapshot[];
  text?: () => Response;
}

/** Serve inference reads and writes with real contract shapes and no worker. */
export function mock_inference_backend(options: BackendOptions = {}) {
  const models = options.models ?? [installed_model];
  const fetch = vi.fn(
    async (url: string | URL | Request, request_options?: RequestInit) => {
      const path = String(url);
      const method = request_options?.method ?? 'GET';
      if (path.endsWith('/health'))
        return Response.json({ status: 'ready', application_version: '0.1.0' });
      if (path.endsWith('/capabilities'))
        return Response.json(
          models.length
            ? capabilities_with_model
            : { ...capabilities_with_model, ...zero_models },
        );
      if (path.endsWith('/workspace')) return Response.json(empty_workspace);
      if (path.endsWith('/models')) return Response.json({ items: models });
      if (path.endsWith('/jobs'))
        return Response.json({ items: options.jobs?.() ?? [] });
      if (path.endsWith('/capacity')) {
        const body = JSON.parse(String(request_options?.body));
        return options.capacity
          ? options.capacity(body)
          : Response.json(capacity_result);
      }
      if (path.endsWith('/decoding_jobs') || path.endsWith('/encoding_jobs'))
        return Response.json(decode_job, { status: 202 });
      if (path.endsWith('/decoded_text') && method === 'DELETE')
        return new Response(null, { status: 204 });
      if (path.endsWith('/decoded_text'))
        return options.text?.() ?? Response.json(decoded_text);
      return Response.json({ items: [] });
    },
  );
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

const zero_models = {
  available_models: [],
  available_profiles: [],
  encoding_available: false,
  decoding_available: false,
  maximum_payload_bytes: 0,
  minimum_image_side: 0,
  maximum_image_side: 0,
};
