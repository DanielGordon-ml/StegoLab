/** Names for raw folders and prepared datasets; the backend DatasetName rule. */
export const SOURCE_NAME_PATTERN = /^[a-z][a-z0-9_-]{0,63}$/;
/** Size of every archive upload part except the last one. */
export const DATASET_UPLOAD_CHUNK_BYTES = 16777216;

export type ArchiveSplitTarget = 'train' | 'tuning' | 'held_out';
export type ArchiveSplits = Record<string, ArchiveSplitTarget> | null;
export type DatasetContent = 'images' | 'text';
export type DatasetSourceKindName =
  | 'uhd_iqa'
  | 'local'
  | 'hugging_face'
  | 'https_archive'
  | 'upload';

/** Request fields with a backend default may be left out; the service fills them. */
export interface HuggingFaceSourceSpec {
  source_kind: 'hugging_face';
  repository: string;
  revision?: string;
  path_prefix?: string | null;
  split?: 'train' | 'validation' | 'test' | null;
  content?: DatasetContent;
  image_column?: string | null;
  text_column?: string | null;
  file_names?: string[] | null;
  maximum_files?: number;
  maximum_download_bytes?: number;
  terms_reference: string;
  archive_splits?: ArchiveSplits;
}
export interface HttpsArchiveSourceSpec {
  source_kind: 'https_archive';
  url: string;
  expected_sha256?: string | null;
  maximum_download_bytes?: number;
  terms_reference: string;
  archive_splits?: ArchiveSplits;
}
export interface UploadSourceSpec {
  source_kind: 'upload';
  upload_identifier: string;
  maximum_download_bytes?: number;
  terms_reference?: string;
  archive_splits?: ArchiveSplits;
}
export type DatasetSourceSpec =
  | HuggingFaceSourceSpec
  | HttpsArchiveSourceSpec
  | UploadSourceSpec;

export interface DatasetInspectionRequest {
  source: DatasetSourceSpec;
  source_name?: string | null;
  maximum_images?: number;
}
export type DatasetAccess =
  | 'available'
  | 'access_required'
  | 'not_found'
  | 'unsupported';
export type RawFolderState = 'available' | 'reusable' | 'conflict';
export interface PlannedAssetRecord {
  path: string;
  sha256: string | null;
  size_bytes: number | null;
}
/** What the backend learned about a source before downloading anything. */
export interface DatasetInspection {
  source_kind: DatasetSourceKindName;
  reference: string;
  requested_revision: string | null;
  resolved_revision: string | null;
  suggested_source_name: string;
  access: DatasetAccess;
  access_guidance: string | null;
  content: DatasetContent;
  asset_count: number;
  assets: PlannedAssetRecord[];
  download_bytes: number | null;
  materialized_bytes: number | null;
  cached_bytes: number;
  supports_pause: boolean;
  declared_splits: boolean;
  materialization_identity: string;
  raw_folder: RawFolderState | null;
  free_disk_bytes: number;
  required_free_bytes: number | null;
  disk_sufficient: boolean | null;
  server_token_configured: boolean;
  warnings: string[];
}

/** Immutable inputs of one download-and-prepare job. */
export interface DatasetFetchRequest {
  client_request_identifier: string;
  operation: 'fetch_dataset';
  source: DatasetSourceSpec;
  source_name: string;
  dataset_name: string;
  maximum_images?: number;
  training_intended?: boolean;
  prepare?: boolean;
}
export type DatasetFetchPhase =
  | 'resolving'
  | 'downloading'
  | 'extracting'
  | 'preparing'
  | 'cancelling'
  | 'pausing'
  | 'paused'
  | 'cleaning'
  | 'completed'
  | 'cancelled';
/** Numbers a fetch job reports while it runs; totals are absent until known. */
export interface DatasetFetchMetrics {
  sequence: number;
  bytes_received: number;
  bytes_total?: number;
  files_completed: number;
  files_total?: number;
  assets_completed: number;
  assets_total: number;
}
export interface TextCorpusSummary {
  bytes: number;
  characters: number;
  files: number;
}
export interface DatasetSummary {
  schema_version: 1;
  dataset_name: string;
  revision: string;
  completion: 'complete';
  integrity: 'not_checked' | 'verified';
  pilot_ready: false;
  reused: boolean;
  full_coverage: boolean;
  expected_images: number | null;
  discovered_images: number;
  selected_images: number;
  accepted_count: number;
  rejection_count: number;
  duplicate_count: number;
  eligible_count: number;
  ineligible_count: number;
  unique_eligible_count: number;
  eligible_by_split: Record<string, number>;
  unique_eligible_by_split: Record<string, number>;
  split_counts: Record<string, number>;
  rejection_reasons: Record<string, number>;
  source_bytes: number;
  prepared_bytes: number;
  warnings: string[];
}
/** Result stored on a completed fetch job. */
export interface DatasetFetchSummary {
  schema_version: 1;
  status: 'completed' | 'stopped' | 'reused';
  source_kind: DatasetSourceKindName;
  source_name: string;
  reference: string;
  resolved_revision: string | null;
  materialization_identity: string;
  raw_folder: string;
  bytes_received: number;
  bytes_total: number | null;
  assets_completed: number;
  assets_total: number;
  member_count: number;
  rejected_member_count: number;
  rejection_reasons: Record<string, number>;
  stop_signal: number | null;
  text_corpus: TextCorpusSummary | null;
  dataset: DatasetSummary | null;
  near_duplicate_audit: 'not_done';
  pilot_ready: false;
  warnings: string[];
}

export interface DatasetUploadCreateRequest {
  client_request_identifier: string;
  file_name: string;
  total_bytes: number;
  expected_sha256?: string | null;
}
/** One archive upload in progress or finished on the backend. */
export interface DatasetUploadSession {
  upload_identifier: string;
  file_name: string;
  total_bytes: number;
  chunk_bytes: 16777216;
  chunk_count: number;
  received_chunks: number[];
  complete: boolean;
  sha256: string | null;
  created_at: string;
  expires_at: string;
}
export interface DatasetUploadChunk {
  upload_identifier: string;
  index: number;
  bytes: number;
  sha256: string;
  received_chunks: number[];
}
export interface DatasetUploadCompleteRequest {
  client_request_identifier: string;
}

/** Disk use of downloads, raw folders and prepared datasets. */
export interface DatasetStorageSummary {
  cache_bytes: number;
  cache_entries: number;
  cache_unused_bytes: number;
  cache_unused_entries: number;
  raw_source_bytes: number;
  raw_source_folders: number;
  prepared_bytes: number;
  prepared_revisions: number;
  free_disk_bytes: number;
  minimum_free_bytes: number;
  active_fetch_jobs: number;
  cleanup_available: boolean;
}
export interface DatasetCacheEntrySummary {
  identity: string;
  kind: 'hugging_face' | 'https_archive' | 'upload';
  reference: string;
  revision: string;
  path: string;
  size_bytes: number;
  created_at: string;
  in_use_by: string[];
}
export interface DatasetCacheSummary {
  root_available: boolean;
  entries: DatasetCacheEntrySummary[];
  total_bytes: number;
  unused_bytes: number;
  unused_entries: number;
  partial_bytes: number;
  partial_entries: number;
  unlisted_entries: number;
}
