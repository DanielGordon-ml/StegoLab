import { useState } from 'react';
import type { Capabilities } from '../contracts/capabilities';
import { Icon } from './icons';

/** Make the inference workflow visible without offering unqualified models. */
export function InferencePanel({
  mode,
  capabilities,
}: {
  mode: 'encode' | 'decode';
  capabilities?: Capabilities;
}) {
  const [message, set_message] = useState('');
  const encoded_bytes = new TextEncoder().encode(message).length;
  const capacity = capabilities?.maximum_payload_bytes ?? 0;
  const encoding = mode === 'encode';
  const reason = capabilities?.available_models.length
    ? 'The inference service and a qualified matching model/profile are required before this workflow can run.'
    : 'No application-ready model is installed. Train and evaluate a model pair first.';
  return (
    <div className="workspace_content">
      <div className="section_heading">
        <span className="eyebrow">
          {encoding ? 'HIDE A MESSAGE' : 'RECOVER A MESSAGE'}
        </span>
        <h1>
          {encoding
            ? 'A message, hidden in plain sight.'
            : 'Bring the message back.'}
        </h1>
        <p>
          {encoding
            ? 'Prepare an image and a private message for verified PNG encoding.'
            : 'Use the original encoded PNG, password, and matching decoder.'}
        </p>
      </div>
      <div className="notice inference_notice">
        <Icon name={mode} />
        <div>
          <strong>
            {encoding
              ? 'Image encoding is not available yet.'
              : 'Image decoding is not available yet.'}
          </strong>
          <p>{reason}</p>
        </div>
      </div>
      <div className="inference_grid">
        <section className="lab_card form_section">
          <label htmlFor={`${mode}_image`}>
            {encoding ? 'Cover image' : 'Encoded PNG'}
          </label>
          <input
            id={`${mode}_image`}
            type="file"
            accept={encoding ? 'image/png,image/jpeg' : 'image/png'}
            disabled
            aria-describedby={`${mode}_reason`}
          />
          <label htmlFor={`${mode}_model`}>
            {encoding ? 'Encoder model' : 'Matching decoder model'}
          </label>
          <select id={`${mode}_model`} disabled>
            <option>No qualified model available</option>
          </select>
          {encoding && (
            <>
              <label htmlFor="secret_message">Message</label>
              <textarea
                id="secret_message"
                rows={5}
                value={message}
                onChange={(event) => set_message(event.target.value)}
                aria-describedby="message_capacity"
                placeholder="Write a message to check its UTF-8 size…"
              />
              <p
                id="message_capacity"
                className={`help_text ${capacity > 0 && encoded_bytes > capacity ? 'invalid_text' : ''}`}
              >
                {encoded_bytes} UTF-8 bytes ·{' '}
                {capacity > 0
                  ? `${capacity} bytes maximum`
                  : 'Capacity is not available until a model is ready'}
              </p>
            </>
          )}
          <label htmlFor={`${mode}_password`}>Password</label>
          <input
            id={`${mode}_password`}
            type="password"
            autoComplete="off"
            disabled
            placeholder="Available with a qualified model"
          />
          <p id={`${mode}_reason`} className="help_text">
            {reason}
          </p>
          <button className="button primary" disabled>
            {encoding ? 'Encode and verify' : 'Authenticate and decode'}
          </button>
        </section>
        <section className="lab_card preview_panel">
          <div className="card_title_row">
            <div>
              <span className="eyebrow">
                {encoding ? 'IMAGE REVIEW' : 'AUTHENTICATED RESULT'}
              </span>
              <h2>{encoding ? 'Before and after' : 'Recovered message'}</h2>
            </div>
            <span className="pill muted">NOT MEASURED</span>
          </div>
          {encoding ? (
            <>
              <div className="image_pair">
                <div>
                  <div className="image_placeholder">
                    <Icon name="encode" />
                  </div>
                  <span>Prepared original</span>
                </div>
                <div>
                  <div className="image_placeholder">
                    <Icon name="encode" />
                  </div>
                  <span>Encoded PNG</span>
                </div>
              </div>
              <p className="empty_text">
                A download is released only after the saved PNG passes message
                recovery verification.
              </p>
              <button className="button secondary" disabled>
                Download verified PNG
              </button>
            </>
          ) : (
            <>
              <div className="decoded_placeholder">
                Your recovered text will appear here after the password and
                message pass authentication.
              </div>
              <div className="form_actions">
                <button className="button secondary" disabled>
                  Copy message
                </button>
                <button className="button secondary" disabled>
                  Clear message
                </button>
              </div>
              <p className="help_text">
                Recovered text clears after five minutes or when you leave
                Decode. Passwords clear after submission.
              </p>
            </>
          )}
        </section>
      </div>
      <p className="configuration_note">
        Keep the PNG unchanged when sharing. Resizing, compression, or editing
        can damage the hidden message. Experimental exports are not approved
        application models.
      </p>
    </div>
  );
}
