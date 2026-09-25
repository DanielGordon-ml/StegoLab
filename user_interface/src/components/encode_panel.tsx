import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { Capabilities } from '../contracts/capabilities';
import type { UploadedImage } from '../contracts/inference';
import {
  is_inference_job,
  latest_job,
  read_capacity,
} from '../contracts/inference_service';
import { validate_encoding_result } from '../contracts/validation';
import { useInferenceJob } from '../hooks/use_inference';
import { is_active_job, type useWorkspace } from '../hooks/use_workspace';
import { ErrorNotice } from './error_notice';
import { ExperimentalBanner } from './experimental_banner';
import { Icon } from './icons';
import { ImagePicker } from './image_picker';
import { InferenceStatus } from './inference_status';
import { ModelChoice } from './model_choice';

const RETENTION_MILLISECONDS = 24 * 60 * 60 * 1000;

/** Hide a message in a cover image and release only a decoder-verified PNG. */
export function EncodePanel({
  state,
  capabilities,
}: {
  state: ReturnType<typeof useWorkspace>;
  capabilities?: Capabilities;
}) {
  const [image, set_image] = useState<UploadedImage | null>(null);
  const [preview, set_preview] = useState<string | null>(null);
  const [model, set_model] = useState('');
  const [message, set_message] = useState('');
  const [password, set_password] = useState('');
  const [download_started, set_download_started] = useState(false);
  const [job_identifier, set_job_identifier] = useState('');
  const [composing, set_composing] = useState(false);
  // Read the clock once per visit; the panel is rebuilt on every tab change.
  const [opened_at] = useState(() => Date.now());
  const jobs = state.jobs.data?.items;
  const job =
    jobs?.find((item) => item.job_identifier === job_identifier) ??
    (composing ? undefined : latest_job(jobs, 'encode'));
  const training_active =
    jobs?.some((item) => is_active_job(item) && !is_inference_job(item)) ??
    false;
  const models_installed = (capabilities?.available_models.length ?? 0) > 0;
  const capacity = useQuery({
    queryKey: ['capacity', image?.image_reference ?? '', model],
    queryFn: () =>
      read_capacity({
        image_reference: image?.image_reference ?? '',
        model_identifier: model,
      }),
    enabled: Boolean(image && model),
  });
  const limit = capacity.data?.maximum_message_bytes;
  const bytes = new TextEncoder().encode(message).length;
  const over = limit != null && bytes > limit;
  const inference = useInferenceJob((started) => {
    set_job_identifier(started.job_identifier);
    set_composing(false);
  });
  const expired =
    job?.status === 'completed' &&
    opened_at - Date.parse(job.updated_at) >= RETENTION_MILLISECONDS;
  const result =
    job?.status === 'completed' &&
    !expired &&
    validate_encoding_result(job.result)
      ? job.result
      : null;
  const original_reference =
    !expired && typeof job?.frozen_settings.image_reference === 'string'
      ? job.frozen_settings.image_reference
      : null;
  const missing = [
    !image && 'a cover image',
    !model && 'a model',
    !password && 'a password',
  ].filter((item): item is string => Boolean(item));
  const ready =
    missing.length === 0 && limit != null && !over && !inference.locked;
  /** Send the frozen request once and clear the password field immediately. */
  function submit() {
    if (!image || !ready) return;
    inference.submit({
      client_request_identifier: crypto.randomUUID(),
      image_reference: image.image_reference,
      model_identifier: model,
      message,
      password,
    });
    set_password('');
    set_download_started(false);
  }
  const artifact = (reference: string) =>
    `/api/v1/artifacts/${encodeURIComponent(reference)}`;
  return (
    <div className="workspace_content">
      <div className="section_heading">
        <span className="eyebrow">HIDE A MESSAGE</span>
        <h1>A message, hidden in plain sight.</h1>
        <p>
          Choose a cover image, write a message, and download a verified PNG.
        </p>
      </div>
      <ExperimentalBanner models_installed={models_installed} mode="encode" />
      <div className="inference_grid">
        <section className="lab_card form_section">
          <ImagePicker
            id="encode_image"
            label="Cover image"
            purpose="cover"
            disabled={inference.locked}
            on_change={(uploaded, url) => {
              set_image(uploaded);
              set_preview(url);
              set_composing(true);
              set_job_identifier('');
              set_download_started(false);
            }}
          />
          <ModelChoice
            id="encode_model"
            label="Encoder model"
            value={model}
            disabled={inference.locked}
            on_change={set_model}
          />
          <label htmlFor="secret_message">Message</label>
          <textarea
            id="secret_message"
            rows={5}
            value={message}
            disabled={inference.locked}
            onChange={(event) => set_message(event.target.value)}
            aria-describedby="message_capacity"
            placeholder="Write the message to hide…"
          />
          <p
            id="message_capacity"
            className={`help_text byte_counter ${over ? 'invalid_text' : ''}`}
          >
            {limit != null
              ? `${bytes} of ${limit} bytes used`
              : `${bytes} bytes · choose an image and a model to see the limit`}
          </p>
          <ErrorNotice error={capacity.error} />
          <label htmlFor="encode_password">Password</label>
          <input
            id="encode_password"
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
            Encode and verify
          </button>
          <p className="help_text" role="status">
            {inference.isPending
              ? 'Starting encode…'
              : missing.length
                ? `To continue, add ${missing.join(', ')}.`
                : over
                  ? 'Shorten the message to fit the limit.'
                  : ''}
          </p>
          <ErrorNotice error={inference.error} />
          {inference.uncertain && (
            <div className="form_actions">
              <button className="button secondary" onClick={inference.retry}>
                Retry encode
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
              <span className="eyebrow">IMAGE REVIEW</span>
              <h2>Before and after</h2>
            </div>
            <span className="pill muted">EXPERIMENTAL</span>
          </div>
          {job && (
            <InferenceStatus
              job={job}
              operation="encode"
              training_active={training_active}
            />
          )}
          <div className="image_pair">
            <div>
              {preview ? (
                <img src={preview} alt="The cover image you chose" />
              ) : original_reference ? (
                <img
                  src={artifact(original_reference)}
                  alt="The prepared cover image of this job"
                />
              ) : (
                <div className="image_placeholder">
                  <Icon name="encode" />
                </div>
              )}
              <span>Prepared original</span>
            </div>
            <div>
              {result ? (
                <img
                  src={artifact(result.artifact_identifier)}
                  alt="The encoded PNG that passed verification"
                />
              ) : (
                <div className="image_placeholder">
                  <Icon name="encode" />
                </div>
              )}
              <span>Encoded PNG</span>
            </div>
          </div>
          {result ? (
            <>
              <p className="help_text result_facts">
                {result.width}×{result.height} px · {result.message_byte_count}{' '}
                message bytes · verified with the matching decoder
              </p>
              <a
                className="button primary"
                href={artifact(result.artifact_identifier)}
                download={result.filename}
                onClick={() => set_download_started(true)}
              >
                Download verified PNG
              </a>
              <p className="help_text" role="status">
                {download_started
                  ? 'Download started. Keep the file unchanged when sharing.'
                  : ''}
              </p>
            </>
          ) : expired ? (
            <p className="empty_text">
              The encoded PNG and its cover were kept for 24 hours and have been
              removed. Encode the image again to download it.
            </p>
          ) : (
            <p className="empty_text">
              A download is released only after the saved PNG passes message
              recovery with the matching decoder.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
