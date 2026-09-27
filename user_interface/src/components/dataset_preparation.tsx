import { useState } from 'react';
import type {
  Capabilities,
  DatasetSourceKind,
} from '../contracts/capabilities';
import type { Workspace } from '../contracts/workspace';
import type { JobSnapshot } from '../contracts/workflows';
import { DatasetSourceFolder } from './dataset_source_folder';
import { DatasetSourceHttps } from './dataset_source_https';
import { DatasetSourceHuggingFace } from './dataset_source_hugging_face';
import { DatasetSourceUpload } from './dataset_source_upload';

const SOURCE_OPTIONS: { kind: DatasetSourceKind; label: string }[] = [
  { kind: 'server_folder', label: 'Server folder' },
  { kind: 'upload', label: 'Upload an archive' },
  { kind: 'hugging_face', label: 'Hugging Face' },
  { kind: 'https_archive', label: 'HTTPS archive' },
];
/** Kinds the contract promises while the capabilities are still being read. */
const DEFAULT_KINDS: DatasetSourceKind[] = ['server_folder'];

/** Choose where the images come from, then fill the form for that source. */
export function DatasetPreparation({
  workspace,
  capabilities,
  on_started,
}: {
  workspace: Workspace;
  capabilities?: Capabilities;
  on_started: (job: JobSnapshot) => void;
}) {
  const [kind, set_kind] = useState<DatasetSourceKind>('server_folder');
  const enabled_kinds = capabilities?.dataset_source_kinds ?? DEFAULT_KINDS;
  return (
    <details className="lab_card preparation_card">
      <summary>
        <span>Prepare a dataset</span>
        <span className="help_text">
          JPEG and PNG · server folders, uploads and downloads
        </span>
      </summary>
      <fieldset className="source_selector">
        <legend>Where are the images?</legend>
        <div className="source_options">
          {SOURCE_OPTIONS.map((option) => {
            const enabled = enabled_kinds.includes(option.kind);
            const note_identifier = `dataset_source_${option.kind}_note`;
            return (
              <div className="source_option" key={option.kind}>
                <label className="source_pill">
                  <input
                    type="radio"
                    name="dataset_source_kind"
                    value={option.kind}
                    checked={kind === option.kind}
                    disabled={!enabled}
                    aria-describedby={enabled ? undefined : note_identifier}
                    onChange={() => set_kind(option.kind)}
                  />
                  <span>{option.label}</span>
                </label>
                {!enabled && (
                  <small id={note_identifier}>Not enabled on this server</small>
                )}
              </div>
            );
          })}
        </div>
      </fieldset>
      {kind === 'server_folder' && (
        <DatasetSourceFolder workspace={workspace} on_started={on_started} />
      )}
      {kind === 'upload' && (
        <DatasetSourceUpload
          capabilities={capabilities}
          on_started={on_started}
        />
      )}
      {kind === 'hugging_face' && (
        <DatasetSourceHuggingFace
          capabilities={capabilities}
          on_started={on_started}
        />
      )}
      {kind === 'https_archive' && (
        <DatasetSourceHttps on_started={on_started} />
      )}
    </details>
  );
}
