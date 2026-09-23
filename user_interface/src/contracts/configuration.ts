/** Persisted defaults for future training runs. */
export interface ConfigurationProfile {
  schema_version: 1;
  checkpoint_interval_seconds: number;
}

/** Immutable save attempt; retain its identifier for uncertain retries. */
export interface ConfigurationUpdate {
  client_request_identifier: string;
  configuration: ConfigurationProfile;
}

/** A reset request uses the same retry rules as a save. */
export interface ConfigurationReset {
  client_request_identifier: string;
}
