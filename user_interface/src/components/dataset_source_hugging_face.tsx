import { useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import type { Capabilities } from '../contracts/capabilities';
import type {
  DatasetContent,
  HuggingFaceSourceSpec,
} from '../contracts/datasets';
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

const REPOSITORY_PATTERN =
  /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const REVISION_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/;
const COLUMN_PATTERN = /^[A-Za-z][A-Za-z0-9_]{0,63}$/;
const SPLITS = ['any', 'train', 'validation', 'test'] as const;
type SplitChoice = (typeof SPLITS)[number];
export const TOKEN_CONFIGURED_TEXT = 'A server token is configured.';
export const TOKEN_MISSING_TEXT =
  'No server token is configured. Public datasets work; gated datasets need a token on the server.';

/** Describe one Hugging Face dataset repository for the backend to download. */
export function DatasetSourceHuggingFace({
  capabilities,
  on_started,
}: {
  capabilities?: Capabilities;
  on_started: (job: JobSnapshot) => void;
}) {
  const [repository, set_repository] = useState('');
  const [revision, set_revision] = useState('main');
  const [path_prefix, set_path_prefix] = useState('');
  const [split, set_split] = useState<SplitChoice>('any');
  const [content, set_content] = useState<DatasetContent>('images');
  const [column, set_column] = useState('');
  const [maximum_images, set_maximum_images] = useState(DEFAULT_MAXIMUM_IMAGES);
  const [names, set_names] = useState<DatasetNames>({
    source_name: '',
    dataset_name: '',
  });
  const [suggested, set_suggested] = useState<string | null>(null);
  const [started, set_started] = useState<JobSnapshot | null>(null);
  const repository_input = useRef<HTMLInputElement>(null);
  const locked = started !== null;
  const repository_text = repository.trim();
  const revision_text = revision.trim();
  const column_text = column.trim();
  const repository_invalid =
    repository_text !== '' && !REPOSITORY_PATTERN.test(repository_text);
  const revision_invalid = !REVISION_PATTERN.test(revision_text);
  const column_invalid =
    column_text !== '' && !COLUMN_PATTERN.test(column_text);
  const images = parse_maximum_images(maximum_images);
  const prepare = content === 'images';
  const source: HuggingFaceSourceSpec | null =
    REPOSITORY_PATTERN.test(repository_text) &&
    !revision_invalid &&
    !column_invalid &&
    images !== null
      ? {
          source_kind: 'hugging_face',
          repository: repository_text,
          revision: revision_text,
          path_prefix: path_prefix.trim() || null,
          split: split === 'any' ? null : split,
          content,
          image_column: prepare && column_text ? column_text : null,
          text_column: !prepare && column_text ? column_text : null,
          terms_reference: `https://huggingface.co/datasets/${repository_text}`,
        }
      : null;
  return (
    <form
      className="form_section dataset_source_form"
      onSubmit={(event) => event.preventDefault()}
    >
      <p className="help_text">
        The backend computer downloads the files. Nothing is sent from this
        browser.
      </p>
      <label htmlFor="hugging_face_repository">Repository</label>
      <input
        id="hugging_face_repository"
        ref={repository_input}
        value={repository}
        disabled={locked}
        placeholder="owner/dataset_name"
        autoComplete="off"
        aria-invalid={repository_invalid || undefined}
        aria-describedby={described_by(
          'hugging_face_repository_help',
          repository_invalid && 'hugging_face_repository_error',
        )}
        onChange={(event) => set_repository(event.target.value)}
      />
      <p id="hugging_face_repository_help" className="help_text">
        The dataset identifier as shown on huggingface.co, for example:
        example/street_photos
      </p>
      {repository_invalid && (
        <p id="hugging_face_repository_error" className="field_error">
          Use the form owner/name with letters, numbers, dots, underscores or
          hyphens.
        </p>
      )}
      <div className="form_columns">
        <div>
          <label htmlFor="hugging_face_revision">Revision</label>
          <input
            id="hugging_face_revision"
            value={revision}
            disabled={locked}
            autoComplete="off"
            aria-invalid={revision_invalid || undefined}
            aria-describedby={described_by(
              'hugging_face_revision_help',
              revision_invalid && 'hugging_face_revision_error',
            )}
            onChange={(event) => set_revision(event.target.value)}
          />
          <p id="hugging_face_revision_help" className="help_text">
            Resolved to an exact commit before download
          </p>
          {revision_invalid && (
            <p id="hugging_face_revision_error" className="field_error">
              Use a branch, tag or commit made of letters, numbers, dots,
              underscores, slashes or hyphens.
            </p>
          )}
        </div>
        <div>
          <label htmlFor="hugging_face_path_prefix">
            Configuration or path prefix (optional)
          </label>
          <input
            id="hugging_face_path_prefix"
            value={path_prefix}
            disabled={locked}
            autoComplete="off"
            placeholder="for example: data/train"
            onChange={(event) => set_path_prefix(event.target.value)}
          />
        </div>
      </div>
      <div className="form_columns">
        <div>
          <label htmlFor="hugging_face_split">Split</label>
          <select
            id="hugging_face_split"
            value={split}
            disabled={locked}
            onChange={(event) => set_split(event.target.value as SplitChoice)}
          >
            {SPLITS.map((choice) => (
              <option key={choice} value={choice}>
                {choice === 'any' ? 'Any split' : choice}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="hugging_face_content">Content</label>
          <select
            id="hugging_face_content"
            value={content}
            disabled={locked}
            onChange={(event) =>
              set_content(event.target.value as DatasetContent)
            }
          >
            <option value="images">Images</option>
            <option value="text">Text</option>
          </select>
        </div>
      </div>
      <div className="form_columns">
        <div>
          <label htmlFor="hugging_face_column">
            {prepare ? 'Image column (optional)' : 'Text column (optional)'}
          </label>
          <input
            id="hugging_face_column"
            value={column}
            disabled={locked}
            autoComplete="off"
            aria-invalid={column_invalid || undefined}
            aria-describedby={described_by(
              column_invalid && 'hugging_face_column_error',
            )}
            onChange={(event) => set_column(event.target.value)}
          />
          {column_invalid && (
            <p id="hugging_face_column_error" className="field_error">
              Start with a letter, then use letters, numbers or underscores.
            </p>
          )}
        </div>
        <MaximumImagesField
          value={maximum_images}
          locked={locked}
          on_change={set_maximum_images}
        />
      </div>
      <p className="help_text token_line">
        {capabilities?.hugging_face_token_configured
          ? TOKEN_CONFIGURED_TEXT
          : TOKEN_MISSING_TEXT}
      </p>
      {!prepare && (
        <p className="help_text">
          Text files are downloaded and kept as a raw folder. No image dataset
          is prepared from them.
        </p>
      )}
      <DatasetNamesFields
        source_name={names.source_name}
        dataset_name={names.dataset_name}
        suggested_source_name={suggested}
        locked={locked}
        show_dataset_name={prepare}
        on_change={set_names}
      />
      <DatasetInspectStep
        source={source}
        source_name={names.source_name}
        dataset_name={names.dataset_name}
        maximum_images={images ?? MAXIMUM_IMAGES_LIMIT}
        training_intended={prepare}
        prepare={prepare}
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
            repository_input.current?.focus();
          }}
        />
      )}
    </form>
  );
}
