import type { DatasetFetchSummary } from '../contracts/datasets';
import { validate_fetch_summary } from '../contracts/dataset_service';
import type { JobSnapshot } from '../contracts/workflows';
import { format_bytes } from './byte_format';

const SOURCE_LABELS: Record<DatasetFetchSummary['source_kind'], string> = {
  uhd_iqa: 'UHD-IQA',
  local: 'Server folder',
  hugging_face: 'Hugging Face',
  https_archive: 'HTTPS archive',
  upload: 'Uploaded archive',
};

/** Combine rejections from extraction and from preparation into one list. */
function rejection_rows(summary: DatasetFetchSummary): [string, number][] {
  const merged: Record<string, number> = { ...summary.rejection_reasons };
  for (const [reason, count] of Object.entries(
    summary.dataset?.rejection_reasons ?? {},
  ))
    merged[reason] = (merged[reason] ?? 0) + count;
  return Object.entries(merged);
}

/** Report what a finished dataset download saved, without any readiness claim. */
export function DatasetSummaryCard({ job }: { job: JobSnapshot }) {
  if (job.operation !== 'fetch_dataset' || !job.result) return null;
  if (!validate_fetch_summary(job.result))
    return (
      <p className="notice">The dataset result format is not supported.</p>
    );
  const summary = job.result;
  const dataset = summary.dataset;
  const checked = summary.member_count;
  const accepted = dataset
    ? dataset.accepted_count
    : checked - summary.rejected_member_count;
  const rejected = Math.max(0, checked - accepted);
  const noun = summary.text_corpus && !dataset ? 'text files' : 'image files';
  const rejections = rejection_rows(summary);
  const warnings = [...summary.warnings, ...(dataset?.warnings ?? [])];
  return (
    <section className="summary_card" aria-label="Dataset download result">
      <span className="eyebrow">DATASET DOWNLOAD · RESULT</span>
      <h3>What was saved</h3>
      <p className="summary_headline">
        {`Checked ${checked} ${noun}: ${accepted} accepted, ${rejected} rejected.`}
      </p>
      {summary.text_corpus && (
        <p className="summary_headline">
          {`Text corpus: ${summary.text_corpus.characters} characters.`}
        </p>
      )}
      <dl className="detail_list">
        <div>
          <dt>Source and revision</dt>
          <dd>
            {[
              SOURCE_LABELS[summary.source_kind],
              summary.reference,
              summary.resolved_revision,
            ]
              .filter(Boolean)
              .join(' · ')}
          </dd>
        </div>
        <div>
          <dt>Raw files folder</dt>
          <dd>{`data/${summary.source_name}`}</dd>
        </div>
        <div>
          <dt>Prepared revision</dt>
          <dd>
            {dataset
              ? `${dataset.dataset_name} · ${dataset.revision.slice(0, 12)}`
              : 'Not prepared'}
          </dd>
        </div>
        <div>
          <dt>Storage</dt>
          <dd>
            {`${format_bytes(summary.bytes_received)} downloaded · ${format_bytes(dataset?.prepared_bytes ?? 0)} prepared`}
          </dd>
        </div>
        <div>
          <dt>Rejection reasons</dt>
          <dd>
            {rejections.length ? (
              <ul>
                {rejections.map(([reason, count]) => (
                  <li key={reason}>{`${reason}: ${count}`}</li>
                ))}
              </ul>
            ) : (
              'None'
            )}
          </dd>
        </div>
      </dl>
      {warnings.length > 0 && (
        <ul className="summary_warnings">
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      <p className="help_text">A near-duplicate audit has not been done.</p>
      <p className="help_text">
        Downloading and preparing files does not make this dataset pilot-ready
        or approved for training.
      </p>
    </section>
  );
}
