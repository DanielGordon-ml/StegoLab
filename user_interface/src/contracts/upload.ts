import { RequestFailure } from './errors';
import type { UploadedImage } from './inference';
import { validate_error, validate_uploaded_image } from './validation';

export const MAXIMUM_UPLOAD_BYTES = 16777216;
export const UPLOAD_TIMEOUT_MILLISECONDS = 60000;
const TOO_LARGE =
  'The image is larger than the 16 MiB upload limit. Choose a smaller file.';

/** Pick a content type the backend accepts, even for files of unknown type. */
function content_type(file: Blob) {
  return file.type === 'image/jpeg' || file.type === 'image/png'
    ? file.type
    : 'application/octet-stream';
}

/** Upload one image as the raw request body, reporting progress and safe failures. */
export function upload_image(
  file: Blob,
  purpose: 'cover' | 'encoded',
  on_progress?: (fraction: number) => void,
): Promise<UploadedImage> {
  if (file.size > MAXIMUM_UPLOAD_BYTES)
    return Promise.reject(new RequestFailure(TOO_LARGE));
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', `/api/v1/images?purpose=${purpose}`);
    request.timeout = UPLOAD_TIMEOUT_MILLISECONDS;
    request.responseType = 'text';
    request.setRequestHeader('Content-Type', content_type(file));
    request.upload.onprogress = (event) => {
      if (event.lengthComputable && on_progress)
        on_progress(event.loaded / event.total);
    };
    request.onerror = () =>
      reject(
        new RequestFailure(
          'Cannot reach StegoLab. Check that the backend is running, then try again.',
          true,
        ),
      );
    request.ontimeout = () =>
      reject(
        new RequestFailure(
          'The upload took too long. Check the connection and try a smaller image.',
          true,
        ),
      );
    request.onabort = () =>
      reject(new RequestFailure('The upload was cancelled.', true));
    request.onload = () => {
      if (request.status === 413) {
        reject(new RequestFailure(TOO_LARGE));
        return;
      }
      let payload: unknown = null;
      try {
        payload = JSON.parse(request.responseText);
      } catch {
        payload = null;
      }
      if (request.status >= 200 && request.status < 300) {
        if (validate_uploaded_image(payload)) resolve(payload);
        else
          reject(
            new RequestFailure(
              'The backend sent an unexpected response. Check the application versions.',
              true,
            ),
          );
        return;
      }
      if (validate_error(payload))
        reject(
          new RequestFailure(
            payload.error.message,
            request.status >= 500,
            payload.error.diagnostic_reference,
          ),
        );
      else
        reject(
          new RequestFailure(
            'The backend could not complete this request. Try again.',
            true,
          ),
        );
    };
    request.send(file);
  });
}
