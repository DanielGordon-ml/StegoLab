import { RequestFailure } from '../contracts/errors';

/** Display only a safe message and its diagnostic reference. */
export function ErrorNotice({ error }: { error: Error | null }) {
  if (!error) return null;
  return (
    <div className="error_notice" role="alert">
      <strong>
        {error instanceof RequestFailure
          ? error.message
          : 'Something went wrong. Please try again.'}
      </strong>
      {error instanceof RequestFailure && error.diagnostic_reference && (
        <span>Reference: {error.diagnostic_reference}</span>
      )}
    </div>
  );
}
