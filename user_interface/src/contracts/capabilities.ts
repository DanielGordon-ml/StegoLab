/** Features and resources confirmed by the running backend. */
export interface Capabilities {
  application_version: string;
  available_devices: string[];
  available_models: string[];
  available_profiles: string[];
  encoding_available: boolean;
  decoding_available: boolean;
  training_available: boolean;
  experimental_models_only: true;
  maximum_payload_bytes: number;
  minimum_image_side: number;
  maximum_image_side: number;
  maximum_upload_bytes: number;
}

/** Backend readiness, separate from model availability. */
export interface HealthStatus {
  status: 'ready';
  application_version: string;
}
