import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import type {
  DatasetInspection,
  DatasetSourceSpec,
} from '../contracts/datasets';
import type { JobSnapshot } from '../contracts/workflows';
import {
  inspect_dataset_source,
  start_dataset_fetch,
} from '../contracts/dataset_service';
import { useWorkflow } from '../hooks/use_workflow';
import { format_bytes } from './byte_format';
import { valid_source_name } from './dataset_source_names';
import { ErrorNotice } from './error_notice';

/** Fields every remote source form hands to the check and fetch steps. */
export interface FetchTarget {
  source: DatasetSourceSpec | null;
  source_name: string;
  dataset_name: string;
  maximum_images: number;
  training_intended: boolean;
  prepare: boolean;
  locked: boolean;
  on_started: (job: JobSnapshot) => void;
}

/** What one check sent, kept so a later field change is noticed. */
interface CheckedRequest {
  source: DatasetSourceSpec;
  source_name: string | null;
  maximum_images: number;
}

const ACCESS_TEXT: Record<
  Exclude<DatasetInspection['access'], 'available'>,
  string
> = {
  access_required: 'Access required. The backend cannot read this source yet.',
  not_found: 'The source was not found. Check the address or repository name.',
  unsupported: 'This kind of source is not supported by the backend.',
};

/** True when the current fields still describe what the backend inspected. */
function still_current(
  checked: CheckedRequest,
  request: CheckedRequest,
  suggested: string,
) {
  return (
    JSON.stringify(checked.source) === JSON.stringify(request.source) &&
    checked.maximum_images === request.maximum_images &&
    (checked.source_name ?? suggested) === (request.source_name ?? suggested)
  );
}

/** Lines of the inspection result in plain words, with byte sizes. */
function result_lines(inspection: DatasetInspection, folder: string) {
  const lines: { text: string; problem?: boolean }[] = [];
  const revision =
    inspection.resolved_revision ??
    inspection.requested_revision ??
    'not available';
  lines.push({ text: `Resolved revision: ${revision}` });
  lines.push({
    text:
      inspection.download_bytes === null
        ? 'Download size unknown — progress will show received bytes only'
        : `Download size: ${format_bytes(inspection.download_bytes)}`,
  });
  const extracted =
    inspection.materialized_bytes === null
      ? ''
      : ` (${format_bytes(inspection.materialized_bytes)})`;
  lines.push({
    text: `Files after extraction: about ${inspection.asset_count}${extracted}`,
  });
  const verdict =
    inspection.disk_sufficient === null
      ? ''
      : inspection.disk_sufficient
        ? ' (enough)'
        : ' (not enough)';
  lines.push({
    text: `Free space: ${format_bytes(inspection.free_disk_bytes)}${verdict}`,
    problem: inspection.disk_sufficient === false,
  });
  if (inspection.required_free_bytes !== null)
    lines.push({
      text: `Space needed: ${format_bytes(inspection.required_free_bytes)}`,
    });
  lines.push({
    text: inspection.supports_pause
      ? 'Pause: available'
      : 'Pause: not available for this source',
  });
  if (inspection.raw_folder === 'available')
    lines.push({ text: `${folder} will be created` });
  if (inspection.raw_folder === 'reusable')
    lines.push({
      text: `${folder} exists with the same source and will be reused`,
    });
  if (inspection.raw_folder === 'conflict')
    lines.push({
      text: `${folder} already holds different files — choose another name`,
      problem: true,
    });
  return lines;
}

/** Show what the backend found, including access problems and warnings. */
export function InspectionResult({
  inspection,
  source_name,
}: {
  inspection: DatasetInspection;
  source_name: string | null;
}) {
  const folder = `data/${source_name ?? inspection.suggested_source_name}`;
  return (
    <div className="inspect_result" role="status">
      {inspection.access !== 'available' && (
        <div className="notice">
          <strong>{ACCESS_TEXT[inspection.access]}</strong>
          {inspection.access_guidance && <p>{inspection.access_guidance}</p>}
        </div>
      )}
      <ul>
        {result_lines(inspection, folder).map((line) => (
          <li key={line.text} className={line.problem ? 'problem' : undefined}>
            {line.text}
          </li>
        ))}
      </ul>
      {inspection.warnings.map((warning) => (
        <p key={warning} className="help_text">
          {warning}
        </p>
      ))}
    </div>
  );
}

/** Why the fetch button stays disabled, or null when the job may start. */
function blocking_reason(
  target: FetchTarget,
  inspection: DatasetInspection | null,
) {
  if (!target.source || !inspection) return 'Check the source first.';
  if (inspection.access !== 'available')
    return 'The backend needs access to this source before it can download.';
  if (inspection.disk_sufficient === false)
    return 'There is not enough free space on the backend computer.';
  if (inspection.raw_folder === 'conflict')
    return 'Choose another raw folder name, then check the source again.';
  if (!valid_source_name(target.source_name))
    return 'Enter a valid raw folder name.';
  if (target.prepare && !valid_source_name(target.dataset_name))
    return 'Enter a valid prepared dataset name.';
  return null;
}

/** Start the frozen download-and-prepare job once every check passed. */
export function DatasetFetchAction({
  inspection,
  ...target
}: FetchTarget & { inspection: DatasetInspection | null }) {
  const workflow = useWorkflow(target.on_started, start_dataset_fetch);
  const reason = blocking_reason(target, inspection);
  function start() {
    if (reason || !target.source) return;
    workflow.submit({
      client_request_identifier: crypto.randomUUID(),
      operation: 'fetch_dataset',
      source: target.source,
      source_name: target.source_name,
      dataset_name: target.prepare ? target.dataset_name : target.source_name,
      maximum_images: target.maximum_images,
      training_intended: target.training_intended,
      prepare: target.prepare,
    });
  }
  return (
    <div className="fetch_action">
      <ErrorNotice error={workflow.error} />
      {workflow.uncertain ? (
        <div className="notice">
          <p>
            Start not confirmed. Retry the same request to find its result
            without starting a second download.
          </p>
          <button
            type="button"
            className="button secondary"
            onClick={workflow.retry}
          >
            Retry fetch
          </button>
        </div>
      ) : (
        <button
          type="button"
          className="button primary"
          disabled={reason !== null || target.locked || workflow.locked}
          onClick={start}
        >
          {workflow.isPending
            ? 'Starting…'
            : target.prepare
              ? 'Fetch and prepare'
              : 'Fetch files'}
        </button>
      )}
      {reason && inspection && <p className="help_text">{reason}</p>}
    </div>
  );
}

/** Check a source before downloading, then start the job from the checked details. */
export function DatasetInspectStep({
  on_inspected,
  ...target
}: FetchTarget & { on_inspected?: (inspection: DatasetInspection) => void }) {
  const [checked, set_checked] = useState<CheckedRequest | null>(null);
  const inspection = useMutation({
    mutationFn: inspect_dataset_source,
    onSuccess: (result) => on_inspected?.(result),
  });
  const request: CheckedRequest | null = target.source
    ? {
        source: target.source,
        source_name: valid_source_name(target.source_name)
          ? target.source_name
          : null,
        maximum_images: target.maximum_images,
      }
    : null;
  const current =
    inspection.data &&
    checked &&
    request &&
    still_current(checked, request, inspection.data.suggested_source_name)
      ? inspection.data
      : null;
  const stale = inspection.data !== undefined && current === null;
  return (
    <div className="inspect_step">
      <div className="form_actions">
        <button
          type="button"
          className="button secondary"
          disabled={!request || target.locked || inspection.isPending}
          onClick={() => {
            if (!request) return;
            set_checked(request);
            inspection.mutate(request);
          }}
        >
          {inspection.isPending ? 'Checking…' : 'Check source'}
        </button>
        {inspection.isPending && (
          <p role="status">Asking the backend what this source contains…</p>
        )}
      </div>
      {stale && (
        <p className="help_text" role="status">
          The source details changed. Check the source again before fetching.
        </p>
      )}
      <ErrorNotice error={inspection.error} />
      {current && (
        <InspectionResult
          inspection={current}
          source_name={checked?.source_name ?? null}
        />
      )}
      <DatasetFetchAction {...target} inspection={current} />
    </div>
  );
}
