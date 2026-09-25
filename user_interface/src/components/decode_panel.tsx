import { useState } from 'react';
import type { Capabilities } from '../contracts/capabilities';
import type { UploadedImage } from '../contracts/inference';
import { is_inference_job, latest_job } from '../contracts/inference_service';
import { useDecodedText } from '../hooks/use_decoded_text';
import { useInferenceJob } from '../hooks/use_inference';
import { is_active_job, type useWorkspace } from '../hooks/use_workspace';
import { ErrorNotice } from './error_notice';
import { ExperimentalBanner } from './experimental_banner';
import { ImagePicker } from './image_picker';
import { InferenceStatus } from './inference_status';
import { ModelChoice } from './model_choice';

/** Recover a message from an unchanged encoded PNG with the matching decoder. */
export function DecodePanel({
  state,
  capabilities,
}: {
  state: ReturnType<typeof useWorkspace>;
  capabilities?: Capabilities;
}) {
  const [image, set_image] = useState<UploadedImage | null>(null);
  const [model, set_model] = useState('');
  const [password, set_password] = useState('');
  const [copy_state, set_copy_state] = useState<'idle' | 'copied' | 'failed'>(
    'idle',
  );
  const [job_identifier, set_job_identifier] = useState('');
  const jobs = state.jobs.data?.items;
  const job =
    jobs?.find((item) => item.job_identifier === job_identifier) ??
    latest_job(jobs, 'decode');
  const training_active =
    jobs?.some((item) => is_active_job(item) && !is_inference_job(item)) ??
    false;
  const models_installed = (capabilities?.available_models.length ?? 0) > 0;
  const inference = useInferenceJob((started) =>
    set_job_identifier(started.job_identifier),
  );
  const decoded = useDecodedText(job);
  const missing = [
    !image && 'the encoded PNG',
    !model && 'a model',
    !password && 'the password',
  ].filter((item): item is string => Boolean(item));
  const ready = missing.length === 0 && !inference.locked;
  /** Send the frozen request once and clear the password field immediately. */
  function submit() {
    if (!image || !ready) return;
    inference.submit({
      client_request_identifier: crypto.randomUUID(),
      image_reference: image.image_reference,
      model_identifier: model,
      password,
    });
    set_password('');
    set_copy_state('idle');
  }
  /** Copy the recovered text without keeping another copy in page state. */
  async function copy() {
    if (!decoded.text) return;
    try {
      await navigator.clipboard.writeText(decoded.text.text);
      set_copy_state('copied');
    } catch {
      set_copy_state('failed');
    }
  }
  /** Forget the text everywhere and reset the copy note with it. */
  async function clear() {
    set_copy_state('idle');
    await decoded.clear();
  }
  return (
    <div className="workspace_content">
      <div className="section_heading">
        <span className="eyebrow">RECOVER A MESSAGE</span>
        <h1>Bring the message back.</h1>
        <p>
          Use the unchanged encoded PNG, its password, and the matching model.
        </p>
      </div>
      <ExperimentalBanner models_installed={models_installed} mode="decode" />
      <div className="inference_grid">
        <section className="lab_card form_section">
          <ImagePicker
            id="decode_image"
            label="Encoded PNG"
            purpose="encoded"
            disabled={inference.locked}
            on_change={(uploaded) => set_image(uploaded)}
          />
          <ModelChoice
            id="decode_model"
            label="Matching decoder model"
            value={model}
            disabled={inference.locked}
            on_change={set_model}
          />
          <label htmlFor="decode_password">Password</label>
          <input
            id="decode_password"
            type="password"
            autoComplete="off"
            value={password}
            disabled={inference.locked}
            maxLength={1024}
            onChange={(event) => set_password(event.target.value)}
          />
          <p className="help_text">
            The password is sent once with the job and never saved.
          </p>
          <button className="button primary" disabled={!ready} onClick={submit}>
            Authenticate and decode
          </button>
          <p className="help_text" role="status">
            {inference.isPending
              ? 'Starting decode…'
              : missing.length
                ? `To continue, add ${missing.join(', ')}.`
                : ''}
          </p>
          <ErrorNotice error={inference.error} />
          {inference.uncertain && (
            <div className="form_actions">
              <button className="button secondary" onClick={inference.retry}>
                Retry decode
              </button>
              <button className="button secondary" onClick={inference.discard}>
                Discard
              </button>
            </div>
          )}
        </section>
        <section className="lab_card preview_panel">
          <div className="card_title_row">
            <div>
              <span className="eyebrow">AUTHENTICATED RESULT</span>
              <h2>Recovered message</h2>
            </div>
            <span className="pill muted">EXPERIMENTAL</span>
          </div>
          {job && (
            <InferenceStatus
              job={job}
              operation="decode"
              training_active={training_active}
            />
          )}
          <div className="decoded_text">
            <label htmlFor="recovered_message">Recovered message</label>
            <textarea
              id="recovered_message"
              rows={7}
              readOnly
              value={decoded.text?.text ?? ''}
              placeholder={
                decoded.loading
                  ? 'Reading the recovered text…'
                  : 'The recovered text appears here after the password and message pass authentication.'
              }
            />
            {decoded.text && (
              <p className="help_text">
                {decoded.text.byte_count} bytes · kept until{' '}
                {new Date(decoded.text.expires_at).toLocaleTimeString()} or
                until you clear it
              </p>
            )}
            <ErrorNotice error={decoded.error} />
          </div>
          <div className="form_actions">
            <button
              className="button secondary"
              disabled={!decoded.text}
              onClick={() => void copy()}
            >
              Copy message
            </button>
            <button
              className="button secondary"
              disabled={!decoded.text}
              onClick={() => void clear()}
            >
              Clear message
            </button>
          </div>
          <p className="help_text" role="status">
            {copy_state === 'copied'
              ? 'Copied to the clipboard.'
              : copy_state === 'failed'
                ? 'Copy failed. Select the text and copy it by hand.'
                : ''}
          </p>
          <p className="help_text">
            Recovered text is forgotten after five minutes, when you clear it,
            or when you leave Decode.
          </p>
        </section>
      </div>
    </div>
  );
}
