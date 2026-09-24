import type { ConfigurationProfile } from './configuration';

/** Immutable inputs for a retry-safe background operation. */
export interface WorkflowRequest {
  client_request_identifier: string;
  operation: 'train' | 'evaluate' | 'export' | 'prepare_dataset';
  experiment_identifier?: string;
  dataset_identifier?: string;
  checkpoint_identifier?: string;
  cpu_threads?: number;
  stop_after_step?: number;
  source_identifier?: string;
  source_subdirectory?: string;
  dataset_name?: string;
  export_name?: string;
  lightweight?: boolean;
}
export interface TrainingPreflight {
  allowed: boolean;
  blockers: string[];
  warnings: string[];
  resolved_settings: Record<string, unknown>;
  remaining_budget_seconds: number | null;
  remaining_experiment_slots: number | null;
}
export interface JobSnapshot {
  job_identifier: string;
  status:
    | 'queued'
    | 'running'
    | 'paused'
    | 'stopped'
    | 'completed'
    | 'cancelled'
    | 'failed'
    | 'interrupted'
    | 'needs_input';
  phase: string;
  configuration: ConfigurationProfile;
  available_actions: string[];
  created_at: string;
  updated_at: string;
  progress: number | null;
  estimated_seconds_remaining: number | null;
  operation?: string | null;
  experiment_identifier?: string | null;
  frozen_settings: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: Record<string, string> | null;
  metrics: Record<string, number>;
  requested_action: string | null;
  latest_event_identifier: number | null;
}
export interface JobList {
  items: JobSnapshot[];
}
export interface JobActionRequest {
  client_request_identifier: string;
  action: 'stop' | 'cancel' | 'pause' | 'resume';
}

export interface JobEvent {
  event_identifier: number;
  job_identifier: string;
  snapshot: JobSnapshot | null;
}
