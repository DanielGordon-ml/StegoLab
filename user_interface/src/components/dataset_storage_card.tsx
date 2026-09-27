import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useRef, useState } from 'react';
import type { DatasetStorageSummary } from '../contracts/datasets';
import {
  read_dataset_storage,
  remove_unused_downloads,
} from '../contracts/dataset_service';
import { format_bytes } from './byte_format';
import { ErrorNotice } from './error_notice';

const STORAGE_QUERY_KEY = ['dataset_storage'];

/** Write a count with the right singular or plural word. */
function count_label(count: number, singular: string, plural: string) {
  return `${count} ${count === 1 ? singular : plural}`;
}

/** Explain why removal is not possible right now, or return null when it is. */
function removal_block(summary: DatasetStorageSummary): string | null {
  if (summary.active_fetch_jobs > 0)
    return 'Removal waits until the running dataset download has finished.';
  if (!summary.cleanup_available)
    return 'There are no unused downloads to remove right now.';
  return null;
}

/** Show where dataset bytes live and remove downloads that no folder uses. */
export function DatasetStorageCard() {
  const client = useQueryClient();
  const [confirming, set_confirming] = useState(false);
  const remove_button = useRef<HTMLButtonElement>(null);
  const storage = useQuery({
    queryKey: STORAGE_QUERY_KEY,
    queryFn: read_dataset_storage,
    staleTime: 30000,
  });
  const removal = useMutation({
    mutationFn: remove_unused_downloads,
    onSuccess: (summary) => {
      client.setQueryData(STORAGE_QUERY_KEY, summary);
    },
    onSettled: () => set_confirming(false),
  });
  const summary = storage.data;
  const block = summary ? removal_block(summary) : null;
  return (
    <section
      className="lab_card storage_card"
      aria-labelledby="dataset_storage_heading"
    >
      <div className="card_title_row">
        <div>
          <span className="eyebrow">DATASET STORAGE</span>
          <h2 id="dataset_storage_heading">Downloads and prepared files</h2>
        </div>
        <span className="pill muted">BACKEND DISK</span>
      </div>
      {storage.isPending && (
        <p className="help_text" role="status">
          Reading dataset storage…
        </p>
      )}
      {storage.isError && (
        <>
          <ErrorNotice error={storage.error} />
          <div className="form_actions">
            <button
              type="button"
              className="button secondary"
              onClick={() => void storage.refetch()}
            >
              Retry reading storage
            </button>
          </div>
        </>
      )}
      {summary && (
        <>
          <dl className="storage_rows">
            <div>
              <dt>Downloaded archives (cache)</dt>
              <dd>
                {`${count_label(summary.cache_entries, 'entry', 'entries')} · ${format_bytes(summary.cache_bytes)} (${format_bytes(summary.cache_unused_bytes)} unused)`}
              </dd>
            </div>
            <div>
              <dt>Raw source folders under data/</dt>
              <dd>
                {`${summary.raw_source_folders} · ${format_bytes(summary.raw_source_bytes)}`}
              </dd>
            </div>
            <div>
              <dt>Prepared datasets</dt>
              <dd>
                {`${count_label(summary.prepared_revisions, 'revision', 'revisions')} · ${format_bytes(summary.prepared_bytes)}`}
              </dd>
            </div>
            <div>
              <dt>Free disk space</dt>
              <dd>
                {`${format_bytes(summary.free_disk_bytes)} (keeps at least ${format_bytes(summary.minimum_free_bytes)} free)`}
              </dd>
            </div>
          </dl>
          {block && <p className="help_text">{block}</p>}
          <ErrorNotice error={removal.error} />
          {removal.isSuccess && (
            <p className="help_text" role="status">
              Unused downloads were removed.
            </p>
          )}
          <div className="form_actions">
            {confirming && (
              <button
                type="button"
                className="button secondary"
                disabled={removal.isPending}
                onClick={() => {
                  set_confirming(false);
                  // This button goes away, so move the keyboard to the one that stays.
                  remove_button.current?.focus();
                }}
              >
                Keep downloads
              </button>
            )}
            <button
              type="button"
              ref={remove_button}
              className={`button ${confirming ? 'primary' : 'secondary'}`}
              disabled={Boolean(block) || removal.isPending}
              onClick={() =>
                confirming ? removal.mutate() : set_confirming(true)
              }
            >
              {removal.isPending
                ? 'Removing…'
                : confirming
                  ? 'Confirm removal'
                  : 'Remove unused downloads'}
            </button>
          </div>
          <p className="help_text">
            Removal only deletes downloaded archives that no raw folder uses.
            Raw folders and prepared datasets stay in place.
          </p>
        </>
      )}
    </section>
  );
}
