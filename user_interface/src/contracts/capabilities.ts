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
  maximum_dataset_upload_bytes: number;
  dataset_upload_chunk_bytes: number;
  hugging_face_token_configured: boolean;
  dataset_source_kinds: DatasetSourceKind[];
}

/** Dataset sources the backend accepts in the Train tab. */
export type DatasetSourceKind =
  | 'server_folder'
  | 'upload'
  | 'hugging_face'
  | 'https_archive';

/** Backend readiness, separate from model availability. */
export interface HealthStatus {
  status: 'ready';
  application_version: string;
}
