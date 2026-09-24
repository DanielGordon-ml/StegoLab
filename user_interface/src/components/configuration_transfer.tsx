import { useRef, useState } from 'react';
import type { ConfigurationProfile } from '../contracts/configuration';
import { validate_configuration } from '../contracts/validation';

/** Exchange only validated default settings, never job data or secret inputs. */
export function ConfigurationTransfer({
  configuration,
  locked,
  on_import,
}: {
  configuration: ConfigurationProfile;
  locked: boolean;
  on_import: (configuration: ConfigurationProfile) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [error, set_error] = useState('');
  /** Validate imported JSON before changing any form state. */
  async function import_file(file?: File) {
    if (!file) return;
    try {
      if (file.size > 16384) throw new Error();
      const value: unknown = JSON.parse(await file.text());
      if (!validate_configuration(value)) throw new Error();
      on_import(value);
      set_error('');
    } catch {
      set_error(
        'This file is not a valid StegoLab configuration. Choose an exported settings JSON file.',
      );
    }
    if (input.current) input.current.value = '';
  }
  /** Download saved settings using a short-lived local object URL. */
  function export_file() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(configuration, null, 2)], {
        type: 'application/json',
      }),
    );
    const link = document.createElement('a');
    link.href = url;
    link.download = 'stegolab-configuration.json';
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <>
      <div className="configuration_transfer">
        <input
          className="visually_hidden"
          aria-label="Import configuration file"
          type="file"
          accept="application/json,.json"
          ref={input}
          disabled={locked}
          onChange={(event) => void import_file(event.target.files?.[0])}
        />
        <button
          className="button secondary"
          disabled={locked}
          onClick={() => input.current?.click()}
        >
          Import configuration
        </button>
        <button className="button secondary" onClick={export_file}>
          Export saved configuration
        </button>
        <span className="help_text">
          Imported values are reviewed before saving.
        </span>
      </div>
      {error && (
        <p className="error_notice" role="alert">
          {error}
        </p>
      )}
    </>
  );
}
