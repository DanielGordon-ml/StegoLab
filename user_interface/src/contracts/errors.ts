/** Safe server error; diagnostic references contain no request content. */
export interface ErrorEnvelope {
  error: {
    code: string;
    message: string;
    diagnostic_reference: string;
  };
}

/** Safe, user-facing request failure without submitted data. */
export class RequestFailure extends Error {
  /** Keep uncertain writes retryable with their original request identifier. */
  constructor(
    message: string,
    readonly uncertain = false,
    readonly diagnostic_reference?: string,
  ) {
    super(message);
    this.name = 'RequestFailure';
  }
}
