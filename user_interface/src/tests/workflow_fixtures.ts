import type { Workspace } from '../contracts/workspace';
import type { JobSnapshot } from '../contracts/workflows';

export const empty_workspace: Workspace = {
  datasets: [],
  runs: [],
  checkpoints: [],
  exports: [],
  sources: [],
  budget: {
    remaining_seconds: 1000,
    remaining_experiments: 2,
    blocked_reason: null,
  },
  warnings: [],
};
export const eligible_dataset = {
  identifier: 'dataset_example',
  name: 'Example images',
  revision: 'revision_1',
  image_count: 12,
  prepared_bytes: 1024,
  training_images: 4,
  tuning_images: 4,
  compatible: true,
  blockers: [],
  integrity: 'not_checked' as const,
};
export const saved_job: JobSnapshot = {
  job_identifier: 'job_example',
  status: 'queued',
  phase: 'waiting_for_worker',
  configuration: { schema_version: 1, checkpoint_interval_seconds: 300 },
  available_actions: ['cancel'],
  created_at: '2026-09-24T10:00:00Z',
  updated_at: '2026-09-24T10:00:00Z',
  progress: null,
  estimated_seconds_remaining: null,
  operation: 'train',
  experiment_identifier: 'example',
  frozen_settings: {},
  result: null,
  error: null,
  metrics: {},
  requested_action: null,
  latest_event_identifier: 1,
};
