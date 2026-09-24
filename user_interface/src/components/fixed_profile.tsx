/** Explain the actual immutable settings of the current CPU profile. */
export function FixedProfile() {
  return (
    <details className="advanced_details">
      <summary>Fixed CPU profile · advanced settings</summary>
      <dl className="detail_list">
        <div>
          <dt>Learning rate</dt>
          <dd>0.0001</dd>
        </div>
        <div>
          <dt>Batch size</dt>
          <dd>1 physical / 4 effective</dd>
        </div>
        <div>
          <dt>Training crop</dt>
          <dd>256 × 256 pixels</dd>
        </div>
        <div>
          <dt>Seed</dt>
          <dd>0</dd>
        </div>
        <div>
          <dt>Validation</dt>
          <dd>Every 100 steps</dd>
        </div>
        <div>
          <dt>Save interval</dt>
          <dd>5 minutes</dd>
        </div>
      </dl>
      <p className="help_text">
        These settings are read-only. Config defaults do not override this
        profile. GPU and custom profiles need a validated runtime and training
        adapter.
      </p>
    </details>
  );
}
