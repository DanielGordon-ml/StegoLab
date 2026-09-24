import type { Workspace } from '../contracts/workspace';

/** Show all prepared datasets while blocking only those this CPU profile cannot use. */
export function DatasetChoice({
  workspace,
  selected,
  locked,
  on_select,
  on_continue,
}: {
  workspace: Workspace;
  selected: string;
  locked: boolean;
  on_select: (identifier: string) => void;
  on_continue: () => void;
}) {
  const dataset = workspace.datasets.find(
    (item) => item.identifier === selected,
  );
  return (
    <div className="form_section">
      <label htmlFor="training_dataset">Prepared dataset</label>
      <select
        id="training_dataset"
        value={selected}
        disabled={locked}
        onChange={(event) => on_select(event.target.value)}
      >
        <option value="">Choose a dataset</option>
        {workspace.datasets.map((item) => (
          <option value={item.identifier} key={item.identifier}>
            {item.name} · {item.image_count} images
            {item.compatible ? '' : ' · unavailable for CPU'}
          </option>
        ))}
      </select>
      <p className="help_text">
        This CPU profile needs four training images of at least 256 × 256 pixels
        and four tuning images of at least 1024 × 1024 pixels. Existing splits
        stay unchanged.
      </p>
      {locked && (
        <p className="help_text">
          Resume uses the original dataset revision and CPU threads.
        </p>
      )}
      {dataset && (
        <>
          <div className="summary_tiles">
            <div>
              <span>Training images</span>
              <strong>{dataset.training_images}</strong>
            </div>
            <div>
              <span>Tuning images</span>
              <strong>{dataset.tuning_images}</strong>
            </div>
            <div>
              <span>Prepared size</span>
              <strong>
                {(dataset.prepared_bytes / 1048576).toFixed(1)} MiB
              </strong>
            </div>
          </div>
          {dataset.blockers.length > 0 && (
            <ul className="blocker_list">
              {dataset.blockers.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          )}
          <p className="help_text">
            File integrity is checked again before starting. Limit: 30 source
            records and 512 MiB.
          </p>
        </>
      )}
      <div className="form_actions">
        <button
          className="button primary"
          disabled={!dataset?.compatible}
          onClick={on_continue}
        >
          Continue to setup →
        </button>
      </div>
    </div>
  );
}
