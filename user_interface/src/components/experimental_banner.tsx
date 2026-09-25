import { Icon } from './icons';

/** State plainly what the experimental model can and cannot promise. */
export function ExperimentalBanner({
  models_installed,
  mode,
}: {
  models_installed: boolean;
  mode: 'encode' | 'decode';
}) {
  return (
    <div className="notice inference_notice" role="note">
      <Icon name={mode} />
      <div>
        {models_installed ? (
          <>
            <strong>Experimental model.</strong>
            <p>
              Encoded images are visibly reduced in quality and message recovery
              is not guaranteed. This is not release quality. Keep the PNG
              unchanged when sharing.
            </p>
          </>
        ) : (
          <>
            <strong>No experimental model is installed.</strong>
            <p>
              Open Train → Model exports and choose Install as experimental
              model.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
