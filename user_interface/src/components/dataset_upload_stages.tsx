import type {
  DatasetInspection,
  DatasetUploadSession,
} from '../contracts/datasets';
import type { JobSnapshot } from '../contracts/workflows';
import { format_byte_progress, format_bytes } from './byte_format';
import { DatasetFetchAction, InspectionResult } from './dataset_inspect_step';
import {
  DatasetNamesFields,
  MAXIMUM_IMAGES_LIMIT,
  valid_source_name,
  type DatasetNames,
} from './dataset_source_names';
import { ErrorNotice } from './error_notice';

/** One finished check of the archive and the raw folder name it was made for. */
export interface CheckedArchive {
  inspection: DatasetInspection;
  source_name: string | null;
}

/** What is known about the archive once every part has reached the backend. */
export interface ArchiveView {
  session: DatasetUploadSession;
  checked: CheckedArchive | null;
  error: Error | null;
  checking: boolean;
}

/** Every state one archive passes through, from choosing it to a started job. */
export type Stage =
  | { kind: 'idle' }
  | {
      kind: 'uploading';
      file: File;
      fraction: number;
      resume_note: string | null;
      controller: AbortController;
    }
  | { kind: 'uploaded'; session: DatasetUploadSession; error: Error | null }
  | { kind: 'inspecting'; session: DatasetUploadSession; first: boolean }
  | {
      kind: 'inspected';
      session: DatasetUploadSession;
      checked: CheckedArchive;
    }
  | { kind: 'started'; job: JobSnapshot }
  | { kind: 'failed'; error: Error };

/** Text of the resume notice, naming the first part that is still missing. */
export function resume_note(session: DatasetUploadSession) {
  const received = new Set(session.received_chunks);
  let part = 0;
  while (received.has(part)) part += 1;
  return `An earlier upload of this file was not finished. Continuing from part ${part + 1} of ${session.chunk_count}.`;
}

/** The raw folder name a check is made for, or null to let the backend suggest one. */
export function name_to_check(source_name: string) {
  return valid_source_name(source_name) ? source_name : null;
}

/** True when the typed raw folder name is no longer the one the backend checked. */
export function name_changed(checked: CheckedArchive, source_name: string) {
  const suggested = checked.inspection.suggested_source_name;
  return (
    (checked.source_name ?? suggested) !==
    (name_to_check(source_name) ?? suggested)
  );
}

/** Turn anything thrown into an Error the notice can show. */
export function as_error(failure: unknown) {
  return failure instanceof Error ? failure : new Error('upload');
}

/** What to show for a fully uploaded archive, or null before the upload is done. */
export function archive_view(stage: Stage): ArchiveView | null {
  const { kind } = stage;
  if (kind === 'uploaded')
    return {
      session: stage.session,
      checked: null,
      error: stage.error,
      checking: false,
    };
  if (kind === 'inspecting' && !stage.first)
    return {
      session: stage.session,
      checked: null,
      error: null,
      checking: true,
    };
  if (kind === 'inspected')
    return {
      session: stage.session,
      checked: stage.checked,
      error: null,
      checking: false,
    };
  return null;
}

export const NAME_CHANGED_TEXT =
  'The raw folder name changed. Check the archive again before fetching.';

/** Progress of the parts on their way to the backend, with a way to stop. */
export function UploadProgress({
  file,
  fraction,
  resume_note,
  on_cancel,
}: {
  file: File;
  fraction: number;
  resume_note: string | null;
  on_cancel: () => void;
}) {
  return (
    <div className="upload_progress">
      <progress aria-label="Archive upload progress" max={1} value={fraction} />
      <p>
        {format_byte_progress(Math.round(fraction * file.size), file.size)} sent
      </p>
      {resume_note && (
        <p className="upload_resume_note" role="status">
          {resume_note}
        </p>
      )}
      <button type="button" className="button secondary" onClick={on_cancel}>
        Cancel upload
      </button>
    </div>
  );
}

/** The uploaded archive, the names to save it under, and the check and fetch steps. */
export function UploadedArchive({
  view,
  names,
  suggested,
  stale,
  on_names_change,
  on_check,
  on_started,
  on_reset,
}: {
  view: ArchiveView;
  names: DatasetNames;
  suggested: string | null;
  stale: boolean;
  on_names_change: (names: DatasetNames) => void;
  on_check: () => void;
  on_started: (job: JobSnapshot) => void;
  on_reset: () => void;
}) {
  const current = view.checked && !stale ? view.checked : null;
  return (
    <>
      <p className="notice success" role="status">
        Upload complete: {view.session.file_name} (
        {format_bytes(view.session.total_bytes)}).
      </p>
      <ErrorNotice error={view.error} />
      <DatasetNamesFields
        source_name={names.source_name}
        dataset_name={names.dataset_name}
        suggested_source_name={suggested}
        locked={false}
        on_change={on_names_change}
      />
      <div className="inspect_step">
        <div className="form_actions">
          <button
            type="button"
            className="button secondary"
            disabled={view.checking}
            onClick={on_check}
          >
            {view.checking ? 'Checking…' : 'Check the archive again'}
          </button>
          {view.checking && <p role="status">Checking the archive again…</p>}
        </div>
        {stale && !view.checking && (
          <p className="help_text" role="status">
            {NAME_CHANGED_TEXT}
          </p>
        )}
        {current && (
          <InspectionResult
            inspection={current.inspection}
            source_name={current.source_name}
          />
        )}
        <DatasetFetchAction
          source={{
            source_kind: 'upload',
            upload_identifier: view.session.upload_identifier,
          }}
          inspection={current?.inspection ?? null}
          source_name={names.source_name}
          dataset_name={names.dataset_name}
          maximum_images={MAXIMUM_IMAGES_LIMIT}
          training_intended
          prepare
          locked={view.checking}
          on_started={on_started}
        />
      </div>
      <button type="button" className="button secondary" onClick={on_reset}>
        Choose a different archive
      </button>
    </>
  );
}
