import type { JobSnapshot } from '../contracts/workflows';
import { SOURCE_NAME_PATTERN } from '../contracts/datasets';

/** The backend keeps at most this many images from one fetch job. */
export const MAXIMUM_IMAGES_LIMIT = 200000;
export const DEFAULT_MAXIMUM_IMAGES = '200000';
export const NAME_RULE =
  'Start with a lowercase letter, then use lowercase letters, numbers, underscores or hyphens, up to 64 characters.';

/** Names typed for the raw folder and for the prepared dataset. */
export interface DatasetNames {
  source_name: string;
  dataset_name: string;
}

/** True when a raw folder or prepared dataset name follows the backend rule. */
export function valid_source_name(name: string) {
  return SOURCE_NAME_PATTERN.test(name);
}

/** Join the identifiers of the help and error texts that describe one field. */
export function described_by(...identifiers: (string | false | null)[]) {
  const present = identifiers.filter((identifier): identifier is string =>
    Boolean(identifier),
  );
  return present.length > 0 ? present.join(' ') : undefined;
}

/** Whole number of images from 1 to the backend limit, or null when invalid. */
export function parse_maximum_images(text: string): number | null {
  const trimmed = text.trim();
  if (!/^[0-9]+$/.test(trimmed)) return null;
  const value = Number(trimmed);
  return value >= 1 && value <= MAXIMUM_IMAGES_LIMIT ? value : null;
}

/** Raw folder and prepared dataset names, with the backend suggestion one click away. */
export function DatasetNamesFields({
  source_name,
  dataset_name,
  suggested_source_name,
  locked,
  on_change,
  show_dataset_name = true,
}: {
  source_name: string;
  dataset_name: string;
  suggested_source_name: string | null;
  locked: boolean;
  on_change: (names: DatasetNames) => void;
  show_dataset_name?: boolean;
}) {
  const source_invalid = source_name !== '' && !valid_source_name(source_name);
  const dataset_invalid =
    dataset_name !== '' && !valid_source_name(dataset_name);
  const suggestion =
    suggested_source_name && suggested_source_name !== source_name
      ? suggested_source_name
      : null;
  const folder = valid_source_name(source_name) ? source_name : '<name>';
  return (
    <div className="dataset_names">
      <div className="form_columns">
        <div>
          <label htmlFor="dataset_source_name">Save raw files as</label>
          <input
            id="dataset_source_name"
            value={source_name}
            disabled={locked}
            maxLength={64}
            autoComplete="off"
            placeholder={suggested_source_name ?? 'for example: street_photos'}
            aria-invalid={source_invalid || undefined}
            aria-describedby={described_by(
              'dataset_source_name_help',
              source_invalid && 'dataset_source_name_error',
            )}
            onChange={(event) =>
              on_change({ source_name: event.target.value, dataset_name })
            }
          />
          <p id="dataset_source_name_help" className="help_text">
            {`Folder data/${folder} on the backend computer.`}
          </p>
          {source_invalid && (
            <p id="dataset_source_name_error" className="field_error">
              {NAME_RULE}
            </p>
          )}
        </div>
        {show_dataset_name && (
          <div>
            <label htmlFor="dataset_prepared_name">Prepared dataset name</label>
            <input
              id="dataset_prepared_name"
              value={dataset_name}
              disabled={locked}
              maxLength={64}
              autoComplete="off"
              aria-invalid={dataset_invalid || undefined}
              aria-describedby={described_by(
                'dataset_prepared_name_help',
                dataset_invalid && 'dataset_prepared_name_error',
              )}
              onChange={(event) =>
                on_change({ source_name, dataset_name: event.target.value })
              }
            />
            <p id="dataset_prepared_name_help" className="help_text">
              Listed under prepared datasets once preparation finishes.
            </p>
            {dataset_invalid && (
              <p id="dataset_prepared_name_error" className="field_error">
                {NAME_RULE}
              </p>
            )}
          </div>
        )}
      </div>
      {suggestion && (
        <button
          type="button"
          className="button secondary"
          disabled={locked}
          onClick={() =>
            on_change({
              source_name: suggestion,
              dataset_name: dataset_name || suggestion,
            })
          }
        >
          Use suggested name: {suggestion}
        </button>
      )}
    </div>
  );
}

/** Upper bound of images one fetch job keeps, from 1 to the backend limit. */
export function MaximumImagesField({
  value,
  locked,
  on_change,
}: {
  value: string;
  locked: boolean;
  on_change: (value: string) => void;
}) {
  const invalid = parse_maximum_images(value) === null;
  return (
    <div>
      <label htmlFor="dataset_maximum_images">Maximum images</label>
      <input
        id="dataset_maximum_images"
        type="number"
        inputMode="numeric"
        min={1}
        max={MAXIMUM_IMAGES_LIMIT}
        step={1}
        value={value}
        disabled={locked}
        aria-invalid={invalid || undefined}
        aria-describedby={described_by(
          invalid && 'dataset_maximum_images_error',
        )}
        onChange={(event) => on_change(event.target.value)}
      />
      {invalid && (
        <p id="dataset_maximum_images_error" className="field_error">
          Enter a whole number between 1 and {MAXIMUM_IMAGES_LIMIT}.
        </p>
      )}
    </div>
  );
}

/** Confirm that the backend accepted a job and let the user set up another one. */
export function StartedNotice({
  job,
  on_reset,
}: {
  job: JobSnapshot;
  on_reset: () => void;
}) {
  return (
    <div className="notice success started_notice" role="status">
      <p>
        Job {job.job_identifier} started. Follow its progress in the job monitor
        above.
      </p>
      <button type="button" className="button secondary" onClick={on_reset}>
        Set up another download
      </button>
    </div>
  );
}
