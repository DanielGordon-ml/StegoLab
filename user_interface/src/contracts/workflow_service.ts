import workspace_schema from '../../../contracts/entities/Workspace.json';
import request_schema from '../../../contracts/entities/WorkflowRequest.json';
import preflight_schema from '../../../contracts/entities/TrainingPreflight.json';
import snapshot_schema from '../../../contracts/entities/JobSnapshot.json';
import list_schema from '../../../contracts/entities/JobList.json';
import event_schema from '../../../contracts/entities/JobEvent.json';
import action_schema from '../../../contracts/entities/WorkflowActionRequest.json';
import { create_validator } from './validation';
import { request_validated } from './transport';
import { RequestFailure } from './errors';
import type { Workspace } from './workspace';
import type {
  WorkflowRequest,
  TrainingPreflight,
  JobSnapshot,
  JobList,
  JobActionRequest,
  JobEvent,
} from './workflows';

export const validate_workspace = create_validator<Workspace>(workspace_schema);
export const validate_workflow =
  create_validator<WorkflowRequest>(request_schema);
export const validate_preflight =
  create_validator<TrainingPreflight>(preflight_schema);
export const validate_job = create_validator<JobSnapshot>(snapshot_schema);
export const validate_job_event = create_validator<JobEvent>(event_schema);
export const validate_jobs = create_validator<JobList>(list_schema);
const validate_action = create_validator<JobActionRequest>(action_schema);

/** Load the live catalog without changing existing assets. */
export const read_workspace = () =>
  request_validated('/workspace', validate_workspace);
/** Load durable jobs after navigation, reconnect, or backend restart. */
export const read_jobs = () => request_validated('/jobs', validate_jobs);
/** Check compatibility without starting a worker or consuming the budget. */
export function check_training(request: WorkflowRequest) {
  const normalized = prepare_request(request);
  return request_validated('/training_preflight', validate_preflight, {
    method: 'POST',
    body: JSON.stringify(normalized),
  });
}
/** Start exactly one registered operation using a stable request identifier. */
export function start_workflow(request: WorkflowRequest) {
  const normalized = prepare_request(request);
  const routes = {
    train: 'training_jobs',
    evaluate: 'evaluation_jobs',
    export: 'export_jobs',
    prepare_dataset: 'dataset_jobs',
  };
  return request_validated(`/${routes[request.operation]}`, validate_job, {
    method: 'POST',
    body: JSON.stringify(normalized),
  });
}
/** Submit an allowed worker action with retry-safe identity. */
export function act_on_job(identifier: string, request: JobActionRequest) {
  const normalized = request;
  if (!validate_action(request))
    throw new RequestFailure('This job action is not supported.');
  return request_validated(
    `/jobs/${encodeURIComponent(identifier)}/actions`,
    validate_job,
    { method: 'POST', body: JSON.stringify(normalized) },
  );
}
/** Validate requests against the backend's exported contract. */
export function prepare_request(request: WorkflowRequest) {
  const defaults = Object.fromEntries(
    Object.entries(request_schema.properties)
      .filter(([, property]) => 'default' in property)
      .map(([name, property]) => [
        name,
        'default' in property ? property.default : undefined,
      ]),
  );
  const normalized = { ...defaults, ...request };
  if (!validate_workflow(normalized))
    throw new RequestFailure('Check the run settings and try again.');
  return normalized;
}
