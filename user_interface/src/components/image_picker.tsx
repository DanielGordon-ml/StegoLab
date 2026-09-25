import { useEffect, useRef, useState, type DragEvent } from 'react';
import type { UploadedImage } from '../contracts/inference';
import { upload_image } from '../contracts/upload';
import { ErrorNotice } from './error_notice';

/** Choose and upload one image, showing a local preview and the backend summary. */
export function ImagePicker({
  id,
  label,
  purpose,
  disabled,
  on_change,
}: {
  id: string;
  label: string;
  purpose: 'cover' | 'encoded';
  disabled?: boolean;
  on_change: (image: UploadedImage | null, preview: string | null) => void;
}) {
  const [preview, set_preview] = useState<string | null>(null);
  const [progress, set_progress] = useState<number | null>(null);
  const [image, set_image] = useState<UploadedImage | null>(null);
  const [error, set_error] = useState<Error | null>(null);
  const [dragging, set_dragging] = useState(false);
  const preview_url = useRef<string | null>(null);
  const generation = useRef(0);
  useEffect(
    () => () => {
      if (preview_url.current) URL.revokeObjectURL(preview_url.current);
    },
    [],
  );
  /** Replace the local preview and release the previous object URL at once. */
  function show_preview(file: File) {
    if (preview_url.current) URL.revokeObjectURL(preview_url.current);
    preview_url.current =
      typeof URL.createObjectURL === 'function'
        ? URL.createObjectURL(file)
        : null;
    set_preview(preview_url.current);
  }
  /** Upload the chosen file and report the accepted image to the owner. */
  async function choose(file: File | undefined) {
    if (!file || disabled || progress !== null) return;
    const attempt = (generation.current += 1);
    const latest = () => attempt === generation.current;
    show_preview(file);
    set_image(null);
    set_error(null);
    set_progress(0);
    on_change(null, preview_url.current);
    try {
      const uploaded = await upload_image(file, purpose, (fraction) => {
        if (latest()) set_progress(fraction);
      });
      if (!latest()) return;
      set_image(uploaded);
      on_change(uploaded, preview_url.current);
    } catch (failure) {
      if (!latest()) return;
      set_error(failure instanceof Error ? failure : new Error('upload'));
      on_change(null, preview_url.current);
    } finally {
      if (latest()) set_progress(null);
    }
  }
  /** Accept a dropped file without letting the browser open it. */
  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    set_dragging(false);
    void choose(event.dataTransfer.files[0]);
  }
  const summary = image?.summary;
  return (
    <>
      <label htmlFor={id}>{label}</label>
      <div
        className={`drop_zone ${dragging ? 'dragging' : ''}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) set_dragging(true);
        }}
        onDragLeave={() => set_dragging(false)}
        onDrop={drop}
      >
        <input
          id={id}
          type="file"
          accept={purpose === 'cover' ? 'image/png,image/jpeg' : 'image/png'}
          disabled={disabled || progress !== null}
          aria-describedby={`${id}_summary`}
          onChange={(event) => {
            const file = event.target.files?.[0];
            // Allow the same file to be chosen again after a failed upload.
            event.target.value = '';
            void choose(file);
          }}
        />
        <p className="help_text">
          {purpose === 'cover'
            ? 'PNG or JPEG up to 16 MiB. Drop a file here or choose one.'
            : 'The unchanged PNG saved by Encode, up to 16 MiB.'}
        </p>
        {preview && (
          <img
            className="picker_preview"
            src={preview}
            alt="The image you chose, before any preparation"
          />
        )}
        {progress !== null && (
          <progress aria-label="Upload progress" max={1} value={progress} />
        )}
      </div>
      <p id={`${id}_summary`} className="help_text">
        {summary
          ? `${summary.prepared_width}×${summary.prepared_height} px · ${summary.source_format} source`
          : progress !== null
            ? 'Uploading and checking the image…'
            : 'No image chosen yet.'}
      </p>
      {image?.warnings.map((warning) => (
        <p key={warning} className="help_text">
          {warning}
        </p>
      ))}
      <ErrorNotice error={error} />
    </>
  );
}
