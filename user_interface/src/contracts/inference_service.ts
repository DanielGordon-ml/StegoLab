import { RequestFailure } from './errors';
import type {
  CapacityRequest,
  DecodingJobRequest,
  EncodingJobRequest,
} from './inference';
import type { ModelInstallRequest } from './models';
import { request_no_content, request_validated } from './transport';
import {
  validate_capacity_result,
  validate_decoded_text,
  validate_decoding_request,
  validate_encoding_request,
  validate_installed_model,
  validate_model_list,
} from './validation';
import { validate_job } from './workflow_service';
import type { JobSnapshot } from './workflows';

/** List the models the backend has explicitly installed for inference. */
export const read_models = () =>
  request_validated('/models', validate_model_list);

/** Install one exported pair as an experimental model with a retry-safe identity. */
export function install_model(request: ModelInstallRequest) {
  return request_validated('/models/install', validate_installed_model, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/** Ask how many message bytes one upload can carry with one model. */
export function read_capacity(request: CapacityRequest) {
  return request_validated('/capacity', validate_capacity_result, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/** Start one encode or decode job; the secrets travel once and are not kept. */
export function start_inference(
  request: EncodingJobRequest | DecodingJobRequest,
) {
  const encoding = 'message' in request;
  const valid = encoding
    ? validate_encoding_request(request)
    : validate_decoding_request(request);
  if (!valid)
    throw new RequestFailure(
      'Check the image, model, message, and password, then try again.',
    );
  return request_validated(
    encoding ? '/encoding_jobs' : '/decoding_jobs',
    validate_job,
    { method: 'POST', body: JSON.stringify(request) },
  );
}

/** Read recovered text from backend memory while it is still available. */
export function read_decoded_text(job_identifier: string) {
  return request_validated(
    `/jobs/${encodeURIComponent(job_identifier)}/decoded_text`,
    validate_decoded_text,
  );
}

/** Ask the backend to forget recovered text; safe to repeat. */
export function discard_decoded_text(job_identifier: string) {
  return request_no_content(
    `/jobs/${encodeURIComponent(job_identifier)}/decoded_text`,
    { method: 'DELETE' },
  );
}

/** Tell encode and decode jobs apart from training-family jobs. */
export function is_inference_job(job: JobSnapshot) {
  return job.operation === 'encode' || job.operation === 'decode';
}

/** Find the most recently created job of one inference operation. */
export function latest_job(
  jobs: JobSnapshot[] | undefined,
  operation: 'encode' | 'decode',
) {
  return [...(jobs ?? [])]
    .filter((job) => job.operation === operation)
    .sort((first, second) => second.created_at.localeCompare(first.created_at))
    .at(0);
}
