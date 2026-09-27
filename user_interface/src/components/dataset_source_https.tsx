import { useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import type { HttpsArchiveSourceSpec } from '../contracts/datasets';
import type { JobSnapshot } from '../contracts/workflows';
import { DatasetInspectStep } from './dataset_inspect_step';
import {
  DEFAULT_MAXIMUM_IMAGES,
  DatasetNamesFields,
  MAXIMUM_IMAGES_LIMIT,
  MaximumImagesField,
  StartedNotice,
  described_by,
  parse_maximum_images,
  type DatasetNames,
} from './dataset_source_names';

const CHECKSUM_PATTERN = /^[a-f0-9]{64}$/;
export const HTTPS_RULE =
  'Use an https:// address that points to a zip or tar archive.';
export const CHECKSUM_RULE =
  'Enter the 64 hexadecimal characters of the SHA-256 digest.';

/** True only for a complete https address; the backend checks it again. */
export function valid_https_address(text: string) {
  try {
    const parsed = new URL(text);
    return parsed.protocol === 'https:' && parsed.hostname !== '';
  } catch {
    return false;
  }
}

/** Describe one archive on a public https address for the backend to download. */
export function DatasetSourceHttps({
  on_started,
}: {
  on_started: (job: JobSnapshot) => void;
}) {
  const [url, set_url] = useState('');
  const [checksum, set_checksum] = useState('');
  const [maximum_images, set_maximum_images] = useState(DEFAULT_MAXIMUM_IMAGES);
  const [names, set_names] = useState<DatasetNames>({
    source_name: '',
    dataset_name: '',
  });
  const [suggested, set_suggested] = useState<string | null>(null);
  const [started, set_started] = useState<JobSnapshot | null>(null);
  const address_input = useRef<HTMLInputElement>(null);
  const locked = started !== null;
  const address = url.trim();
  const digest = checksum.trim().toLowerCase();
  const address_invalid = address !== '' && !valid_https_address(address);
  const checksum_invalid = digest !== '' && !CHECKSUM_PATTERN.test(digest);
  const images = parse_maximum_images(maximum_images);
  const source: HttpsArchiveSourceSpec | null =
    valid_https_address(address) && !checksum_invalid && images !== null
      ? {
          source_kind: 'https_archive',
          url: address,
          expected_sha256: digest || null,
          terms_reference: address,
        }
      : null;
  return (
    <form
      className="form_section dataset_source_form"
      onSubmit={(event) => event.preventDefault()}
    >
      <p className="help_text">
        The backend computer downloads one zip or tar archive from a public
        https address and unpacks it. Nothing is sent from this browser.
      </p>
      <label htmlFor="https_archive_url">Archive address</label>
      <input
        id="https_archive_url"
        ref={address_input}
        type="url"
        inputMode="url"
        value={url}
        disabled={locked}
        autoComplete="off"
        placeholder="https://example.org/images.zip"
        aria-invalid={address_invalid || undefined}
        aria-describedby={described_by(
          address_invalid && 'https_archive_url_error',
        )}
        onChange={(event) => set_url(event.target.value)}
      />
      {address_invalid && (
        <p id="https_archive_url_error" className="field_error">
          {HTTPS_RULE}
        </p>
      )}
      <label htmlFor="https_archive_checksum">
        Expected SHA-256 (optional)
      </label>
      <input
        id="https_archive_checksum"
        value={checksum}
        disabled={locked}
        autoComplete="off"
        maxLength={64}
        aria-invalid={checksum_invalid || undefined}
        aria-describedby={described_by(
          'https_archive_checksum_help',
          checksum_invalid && 'https_archive_checksum_error',
        )}
        onChange={(event) => set_checksum(event.target.value)}
      />
      <p id="https_archive_checksum_help" className="help_text">
        When given, the download is refused if its digest differs.
      </p>
      {checksum_invalid && (
        <p id="https_archive_checksum_error" className="field_error">
          {CHECKSUM_RULE}
        </p>
      )}
      <div className="form_columns">
        <MaximumImagesField
          value={maximum_images}
          locked={locked}
          on_change={set_maximum_images}
        />
      </div>
      <DatasetNamesFields
        source_name={names.source_name}
        dataset_name={names.dataset_name}
        suggested_source_name={suggested}
        locked={locked}
        on_change={set_names}
      />
      <DatasetInspectStep
        source={source}
        source_name={names.source_name}
        dataset_name={names.dataset_name}
        maximum_images={images ?? MAXIMUM_IMAGES_LIMIT}
        training_intended
        prepare
        locked={locked}
        on_inspected={(inspection) => {
          set_suggested(inspection.suggested_source_name);
          set_names((previous) => ({
            source_name:
              previous.source_name || inspection.suggested_source_name,
            dataset_name:
              previous.dataset_name || inspection.suggested_source_name,
          }));
        }}
        on_started={(job) => {
          set_started(job);
          on_started(job);
        }}
      />
      {started && (
        <StartedNotice
          job={started}
          on_reset={() => {
            // Render the unlocked fields first so the keyboard can land on one.
            flushSync(() => set_started(null));
            address_input.current?.focus();
          }}
        />
      )}
    </form>
  );
}
