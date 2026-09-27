import { useRef, useState, type DragEvent } from 'react';
import { flushSync } from 'react-dom';
import type { Capabilities } from '../contracts/capabilities';
import type { DatasetUploadSession } from '../contracts/datasets';
import type { JobSnapshot } from '../contracts/workflows';
import { inspect_dataset_source } from '../contracts/dataset_service';
import {
  read_remembered_upload,
  upload_dataset_archive,
} from '../contracts/dataset_upload_flow';
import { RequestFailure } from '../contracts/errors';
import { format_bytes } from './byte_format';
import { StartedNotice, type DatasetNames } from './dataset_source_names';
import {
  UploadProgress,
  UploadedArchive,
  archive_view,
  as_error,
  name_changed,
  name_to_check,
  resume_note,
  type Stage,
} from './dataset_upload_stages';
import { ErrorNotice } from './error_notice';

const DEFAULT_MAXIMUM_UPLOAD_BYTES = 2147483648;
const ACCEPTED_FILES = '.zip,.tar,.tar.gz,application/zip';

/** Upload one archive in parts, let the backend check it, then fetch from it. */
export function DatasetSourceUpload({
  capabilities,
  on_started,
}: {
  capabilities?: Capabilities;
  on_started: (job: JobSnapshot) => void;
}) {
  const [stage, set_stage] = useState<Stage>({ kind: 'idle' });
  const [remembered, set_remembered] = useState(read_remembered_upload);
  const [names, set_names] = useState<DatasetNames>({
    source_name: '',
    dataset_name: '',
  });
  const [suggested, set_suggested] = useState<string | null>(null);
  const [dragging, set_dragging] = useState(false);
  const generation = useRef(0);
  const stage_region = useRef<HTMLDivElement>(null);
  const file_input = useRef<HTMLInputElement>(null);
  const maximum_bytes =
    capabilities?.maximum_dataset_upload_bytes ?? DEFAULT_MAXIMUM_UPLOAD_BYTES;
  const busy = stage.kind === 'uploading' || stage.kind === 'inspecting';
  const view = archive_view(stage);
  const stale = view?.checked
    ? name_changed(view.checked, names.source_name)
    : false;
  /** Keep keyboard focus inside the form when the control that held it goes away. */
  function focus_stage() {
    stage_region.current?.focus();
  }
  /** Begin a new attempt; answers that belong to older attempts are ignored. */
  function begin_attempt() {
    const attempt = (generation.current += 1);
    return () => attempt === generation.current;
  }
  /** Ask the backend what the archive holds for one raw folder name. */
  async function inspect(
    session: DatasetUploadSession,
    source_name: string | null,
    first: boolean,
  ) {
    const latest = begin_attempt();
    set_stage({ kind: 'inspecting', session, first });
    try {
      const inspection = await inspect_dataset_source({
        source: {
          source_kind: 'upload',
          upload_identifier: session.upload_identifier,
        },
        source_name,
      });
      if (!latest()) return;
      set_suggested(inspection.suggested_source_name);
      set_names((previous) => ({
        source_name: previous.source_name || inspection.suggested_source_name,
        dataset_name: previous.dataset_name || inspection.suggested_source_name,
      }));
      set_stage({
        kind: 'inspected',
        session,
        checked: { inspection, source_name },
      });
    } catch (failure) {
      if (latest())
        set_stage({ kind: 'uploaded', session, error: as_error(failure) });
    }
  }
  /** Send the archive, then ask the backend what it contains, in one flow. */
  async function choose(file: File | undefined) {
    if (!file || busy) return;
    const latest = begin_attempt();
    if (file.size > maximum_bytes) {
      set_stage({
        kind: 'failed',
        error: new RequestFailure(
          `This archive is ${format_bytes(file.size)}. The server accepts archives up to ${format_bytes(maximum_bytes)}.`,
        ),
      });
      return;
    }
    const controller = new AbortController();
    set_stage({
      kind: 'uploading',
      file,
      fraction: 0,
      resume_note: null,
      controller,
    });
    focus_stage();
    let session: DatasetUploadSession | null = null;
    try {
      session = await upload_dataset_archive(file, {
        signal: controller.signal,
        on_progress: (fraction) => {
          if (latest())
            set_stage((previous) =>
              previous.kind === 'uploading'
                ? { ...previous, fraction }
                : previous,
            );
        },
        on_session: (found, resumed) => {
          if (latest() && resumed)
            set_stage((previous) =>
              previous.kind === 'uploading'
                ? { ...previous, resume_note: resume_note(found) }
                : previous,
            );
        },
      });
    } catch (failure) {
      if (latest()) set_stage({ kind: 'failed', error: as_error(failure) });
    }
    if (!latest()) return;
    // The flow forgets a finished, expired or discarded session as it goes.
    set_remembered(read_remembered_upload());
    if (!session) return;
    set_stage({ kind: 'uploaded', session, error: null });
    await inspect(session, name_to_check(names.source_name), true);
  }
  /** Forget the current archive and put the keyboard back on the file input. */
  function reset() {
    generation.current += 1;
    flushSync(() => set_stage({ kind: 'idle' }));
    file_input.current?.focus();
  }
  /** Accept a dropped archive without letting the browser open it. */
  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    set_dragging(false);
    void choose(event.dataTransfer.files[0]);
  }
  return (
    <form
      className="form_section dataset_source_form"
      onSubmit={(event) => event.preventDefault()}
    >
      <p className="help_text">
        Send one zip or tar archive of JPEG or PNG files from this computer, up
        to {format_bytes(maximum_bytes)}. It travels in 16 MiB parts and can
        continue after an interruption.
      </p>
      {remembered && (stage.kind === 'idle' || stage.kind === 'failed') && (
        <p className="upload_resume_note" role="status">
          An earlier upload of {remembered.file_name} was not finished. Choose
          the same file to continue where it stopped.
        </p>
      )}
      <div
        className="upload_stage"
        role="group"
        aria-label="Archive upload"
        tabIndex={-1}
        ref={stage_region}
      >
        {(stage.kind === 'idle' || stage.kind === 'failed') && (
          <>
            <label htmlFor="dataset_archive">Archive file</label>
            <div
              className={`drop_zone ${dragging ? 'dragging' : ''}`}
              onDragOver={(event) => {
                event.preventDefault();
                set_dragging(true);
              }}
              onDragLeave={() => set_dragging(false)}
              onDrop={drop}
            >
              <input
                id="dataset_archive"
                ref={file_input}
                type="file"
                accept={ACCEPTED_FILES}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  // Allow the same file to be chosen again after a failure.
                  event.target.value = '';
                  void choose(file);
                }}
              />
              <p className="help_text">
                Drop a .zip, .tar or .tar.gz file here or choose one.
              </p>
            </div>
          </>
        )}
        {stage.kind === 'uploading' && (
          <UploadProgress
            file={stage.file}
            fraction={stage.fraction}
            resume_note={stage.resume_note}
            on_cancel={() => {
              stage.controller.abort();
              focus_stage();
            }}
          />
        )}
        {stage.kind === 'inspecting' && stage.first && (
          <p role="status">Upload complete. Checking the archive…</p>
        )}
        {stage.kind === 'failed' && <ErrorNotice error={stage.error} />}
        {view && (
          <UploadedArchive
            view={view}
            names={names}
            suggested={suggested}
            stale={stale}
            on_names_change={set_names}
            on_check={() =>
              void inspect(
                view.session,
                name_to_check(names.source_name),
                false,
              )
            }
            on_started={(job) => {
              set_stage({ kind: 'started', job });
              on_started(job);
            }}
            on_reset={reset}
          />
        )}
        {stage.kind === 'started' && (
          <StartedNotice job={stage.job} on_reset={reset} />
        )}
      </div>
    </form>
  );
}
