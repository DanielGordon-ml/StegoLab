import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import type {
  ConfigurationProfile,
  ConfigurationReset,
  ConfigurationUpdate,
} from '../contracts/configuration';
import { RequestFailure } from '../contracts/errors';
import {
  read_configuration,
  reset_configuration,
  save_configuration,
} from '../contracts/service';
import { ErrorNotice } from './error_notice';

type Change =
  | { kind: 'save'; request: ConfigurationUpdate }
  | { kind: 'reset'; request: ConfigurationReset };

/** Convert minutes without rounding, truncating, or accepting partial input. */
export function parse_minutes(value: string): number | null {
  if (!/^[1-9][0-9]*$/.test(value)) return null;
  const seconds = Number(value) * 60;
  return Number.isSafeInteger(seconds) ? seconds : null;
}

/** Fetch saved defaults while preserving the form through later read failures. */
export function ConfigurationPanel() {
  const configuration = useQuery({
    queryKey: ['configuration'],
    queryFn: read_configuration,
  });
  return (
    <div className="workspace_content">
      <div className="section_heading">
        <span className="eyebrow">MAKE IT YOURS</span>
        <h1>Workspace settings</h1>
        <p>A few simple defaults. Saved locally, ready for your next visit.</p>
      </div>
      {configuration.isPending && (
        <p role="status" className="loading_card">
          Loading your saved settings…
        </p>
      )}
      {configuration.isError && (
        <>
          <ErrorNotice error={configuration.error} />
          <button
            className="button secondary"
            onClick={() => void configuration.refetch()}
          >
            Retry loading settings
          </button>
        </>
      )}
      {configuration.data && (
        <ConfigurationForm initial_configuration={configuration.data} />
      )}
    </div>
  );
}

/** Save defaults only after confirmation, reusing the exact request on uncertain retries. */
function ConfigurationForm({
  initial_configuration,
}: {
  initial_configuration: ConfigurationProfile;
}) {
  const query_client = useQueryClient();
  const [minutes, set_minutes] = useState(
    String(initial_configuration.checkpoint_interval_seconds / 60),
  );
  const [confirmation, set_confirmation] = useState('');
  const [attempt, set_attempt] = useState<Change | null>(null);
  const mutation = useMutation({
    mutationFn: (change: Change) =>
      change.kind === 'save'
        ? save_configuration(change.request)
        : reset_configuration(change.request),
    onMutate: async () => {
      await query_client.cancelQueries({ queryKey: ['configuration'] });
    },
    onSuccess: async (configuration, change) => {
      await query_client.cancelQueries({ queryKey: ['configuration'] });
      query_client.setQueryData(['configuration'], configuration);
      set_minutes(String(configuration.checkpoint_interval_seconds / 60));
      set_confirmation(
        change.kind === 'reset'
          ? 'Defaults restored and saved.'
          : 'Settings saved.',
      );
      set_attempt(null);
    },
  });
  const seconds = parse_minutes(minutes);
  const uncertain =
    mutation.isError &&
    mutation.error instanceof RequestFailure &&
    mutation.error.uncertain;
  const locked = mutation.isPending || uncertain;
  const dirty = seconds !== initial_configuration.checkpoint_interval_seconds;

  /** Freeze the submitted value and request identifier before starting a write. */
  function submit_change(kind: 'save' | 'reset') {
    if (kind === 'save' && seconds === null) return;
    const identifier = crypto.randomUUID();
    const change: Change =
      kind === 'save'
        ? {
            kind,
            request: {
              client_request_identifier: identifier,
              configuration: {
                schema_version: 1,
                checkpoint_interval_seconds: seconds!,
              },
            },
          }
        : { kind, request: { client_request_identifier: identifier } };
    set_confirmation('');
    set_attempt(change);
    mutation.mutate(change);
  }

  return (
    <>
      <form
        className="settings_card"
        onSubmit={(event) => {
          event.preventDefault();
          submit_change('save');
        }}
      >
        <div className="card_heading">
          <div>
            <span className="eyebrow">TRAINING DEFAULTS</span>
            <h2>Keep your progress</h2>
          </div>
          <span className="pill muted">FOR FUTURE RUNS</span>
        </div>
        <div className="settings_field">
          <div>
            <label htmlFor="checkpoint_minutes">Checkpoint frequency</label>
            <p id="checkpoint_help">
              How often a future training run should save its progress.
              <br />
              Training is not available in this release.
            </p>
          </div>
          <div className="number_field">
            <input
              id="checkpoint_minutes"
              name="checkpoint_minutes"
              type="number"
              min="1"
              step="1"
              inputMode="numeric"
              value={minutes}
              disabled={locked}
              aria-invalid={seconds === null}
              aria-describedby={`checkpoint_help${seconds === null ? ' checkpoint_error' : ''}`}
              onChange={(event) => {
                set_minutes(event.target.value);
                set_confirmation('');
                mutation.reset();
              }}
            />
            <span>minutes</span>
          </div>
        </div>
        {seconds === null && (
          <p className="field_error" id="checkpoint_error">
            Enter a positive whole number of minutes.
          </p>
        )}
        <div className="settings_actions">
          <button
            type="button"
            className="button secondary"
            disabled={locked}
            onClick={() => submit_change('reset')}
          >
            Reset to defaults
          </button>
          <span className="saved_status" role="status">
            {mutation.isPending
              ? 'Saving settings…'
              : confirmation || (dirty ? 'Unsaved changes' : 'Up to date')}
          </span>
          <button
            type="submit"
            className="button primary"
            disabled={locked || seconds === null || !dirty}
          >
            Save settings<span aria-hidden="true"> ↗</span>
          </button>
        </div>
      </form>
      <ErrorNotice error={mutation.error} />
      {uncertain && attempt && (
        <div className="retry_row">
          <p>
            Save not confirmed. Retry the same request to check its result. Your
            entered setting is kept.
          </p>
          <button
            className="button secondary"
            onClick={() => mutation.mutate(attempt)}
          >
            Retry save
          </button>
        </div>
      )}
      <p className="configuration_note">
        Changes are saved on this computer. The default frequency is 5 minutes.
      </p>
    </>
  );
}
