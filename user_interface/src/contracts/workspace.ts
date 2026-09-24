/** Registered local assets; identifiers never stand in for arbitrary paths. */
export interface QualityMetrics {
  exact_recovery_rate: number | null;
  trial_count: number | null;
  psnr: number | null;
  psnr_sample_count: number | null;
  ssim: number | null;
  ssim_sample_count: number | null;
}
export interface WorkspaceDataset {
  identifier: string;
  name: string;
  revision: string;
  image_count: number;
  prepared_bytes: number;
  training_images: number;
  tuning_images: number;
  compatible: boolean;
  blockers: string[];
  integrity: 'not_checked' | 'verified';
}
export interface WorkspaceRun {
  identifier: string;
  experiment_identifier: string;
  status: string;
  global_step: number;
  dataset_revision: string | null;
  created_at: string;
  metrics: QualityMetrics;
}
export interface WorkspaceCheckpoint {
  dataset_revision: string;
  cpu_threads: number | null;
  identifier: string;
  experiment_identifier: string;
  global_step: number;
  created_at: string;
  latest: boolean;
  pinned: boolean;
  metrics: QualityMetrics;
  resume_blockers: string[];
}
export interface Workspace {
  datasets: WorkspaceDataset[];
  runs: WorkspaceRun[];
  checkpoints: WorkspaceCheckpoint[];
  exports: {
    identifier: string;
    name: string;
    kind: 'encoder' | 'decoder';
    experimental: true;
    artifact_identifier: string;
  }[];
  sources: { identifier: string; label: string }[];
  budget: {
    remaining_seconds: number | null;
    remaining_experiments: number | null;
    blocked_reason: string | null;
  };
  warnings: string[];
}
