import { useState } from 'react';
import type { Workspace } from '../contracts/workspace';
import type { JobSnapshot } from '../contracts/workflows';
import { useWorkflow } from '../hooks/use_workflow';
import { ErrorNotice } from './error_notice';

/** Prepare images from a folder registered on the backend computer. */
export function DatasetSourceFolder({
  workspace,
  on_started,
}: {
  workspace: Workspace;
  on_started: (job: JobSnapshot) => void;
}) {
  const [source, set_source] = useState('');
  const [subdirectory, set_subdirectory] = useState('');
  const [name, set_name] = useState('local_images');
  const workflow = useWorkflow(on_started);
  const valid = Boolean(source) && /^[a-z][a-z0-9_]{0,63}$/.test(name);
  const folders =
    workspace.sources.find((item) => item.identifier === source)?.folders ?? [];
  return (
    <form
      className="form_section"
      onSubmit={(event) => {
        event.preventDefault();
        if (valid)
          workflow.submit({
            client_request_identifier: crypto.randomUUID(),
            operation: 'prepare_dataset',
            source_identifier: source,
            source_subdirectory: subdirectory,
            dataset_name: name,
          });
      }}
    >
      <p className="help_text">
        These folders belong to the computer running the backend. They are not
        folders on your browser's computer.
      </p>
      <label htmlFor="source_folder">Source folder</label>
      <select
        id="source_folder"
        value={source}
        disabled={workflow.locked}
        onChange={(event) => set_source(event.target.value)}
      >
        <option value="">Choose a registered folder</option>
        {workspace.sources.map((item) => (
          <option key={item.identifier} value={item.identifier}>
            {item.label}
          </option>
        ))}
      </select>
      {workspace.sources.length === 0 && (
        <p className="notice">
          No source folders are registered. Configure a local source on the
          backend first.
        </p>
      )}
      <div className="form_columns">
        <div>
          <label htmlFor="source_subdirectory">Subfolder (optional)</label>
          <input
            id="source_subdirectory"
            list="source_subdirectory_options"
            value={subdirectory}
            disabled={workflow.locked}
            onChange={(event) => set_subdirectory(event.target.value)}
            placeholder="for example: landscape"
          />
          <datalist id="source_subdirectory_options">
            {folders.map((folder) => (
              <option key={folder} value={folder} />
            ))}
          </datalist>
        </div>
        <div>
          <label htmlFor="dataset_name">Dataset name</label>
          <input
            id="dataset_name"
            value={name}
            disabled={workflow.locked}
            maxLength={64}
            pattern="[a-z][a-z0-9_]{0,63}"
            onChange={(event) => set_name(event.target.value)}
          />
        </div>
      </div>
      <p className="help_text">
        Keep the CPU dataset within 30 source records and 512 MiB. Training
        requires four training images of at least 256 × 256 pixels and four
        tuning images of at least 1024 × 1024 pixels. Preparation reports
        rejected files.
      </p>
      <ErrorNotice error={workflow.error} />
      {workflow.uncertain ? (
        <button
          type="button"
          className="button secondary"
          onClick={workflow.retry}
        >
          Retry preparation
        </button>
      ) : (
        <button className="button primary" disabled={!valid || workflow.locked}>
          {workflow.isPending ? 'Starting preparation…' : 'Prepare images'}
        </button>
      )}
    </form>
  );
}
