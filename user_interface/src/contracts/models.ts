/** One explicitly installed experimental encoder/decoder pair. */
export interface InstalledModel {
  model_identifier: string;
  compatibility_identifier: string;
  source_identifier: string;
  profile_identifier: 'test_only_v1';
  status: 'experimental';
  format_version: number;
  minimum_side: number;
  maximum_side: number;
  maximum_payload_bytes: number;
  installed_at: string;
  export_reference: string;
}

/** Installed models advertised by the backend. */
export interface ModelList {
  items: InstalledModel[];
}

/** Explicit installation request with a retry-safe identifier. */
export interface ModelInstallRequest {
  client_request_identifier: string;
  export_reference: string;
}
