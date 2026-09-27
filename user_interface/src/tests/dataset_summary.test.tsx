import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { DatasetSummaryCard } from '../components/dataset_summary_card';
import type { JobSnapshot } from '../contracts/workflows';
import { fetch_summary, saved_job } from './workflow_fixtures';

const AUDIT_NOTE = 'A near-duplicate audit has not been done.';
const APPROVAL_NOTE =
  'Downloading and preparing files does not make this dataset pilot-ready or approved for training.';

/** Mount the card for a completed download with the given result. */
function render_card(result: JobSnapshot['result']) {
  return render(
    <DatasetSummaryCard
      job={{
        ...saved_job,
        operation: 'fetch_dataset',
        status: 'completed',
        phase: 'completed',
        available_actions: [],
        result,
      }}
    />,
  );
}

describe('dataset summary card', () => {
  it('shows counts, folders and revision with both fixed notes', () => {
    const { container } = render_card(fetch_summary);
    expect(
      screen.getByText('Checked 3 image files: 2 accepted, 1 rejected.'),
    ).toBeVisible();
    expect(screen.getByText('data/example_source')).toBeVisible();
    expect(screen.getByText('example_prepared · abcdef123456')).toBeVisible();
    expect(
      screen.getByText('Uploaded archive · example_images.zip'),
    ).toBeVisible();
    expect(
      screen.getByText('12.0 MB downloaded · 9.0 MB prepared'),
    ).toBeVisible();
    expect(screen.getByText('not_an_image: 1')).toBeVisible();
    expect(screen.getByText(AUDIT_NOTE)).toBeVisible();
    expect(screen.getByText(APPROVAL_NOTE)).toBeVisible();
    expect(container.querySelectorAll('.pill, .status_pill')).toHaveLength(0);
    expect(container.textContent).not.toMatch(/compatible/i);
    expect(container.textContent?.match(/pilot-ready/g)).toHaveLength(1);
  });
  it('reports a text corpus without a prepared dataset', () => {
    render_card({
      ...fetch_summary,
      dataset: null,
      text_corpus: { files: 2, characters: 1200, bytes: 1300 },
    });
    expect(
      screen.getByText('Checked 3 text files: 2 accepted, 1 rejected.'),
    ).toBeVisible();
    expect(screen.getByText('Text corpus: 1200 characters.')).toBeVisible();
    expect(screen.getByText('Not prepared')).toBeVisible();
    expect(screen.getByText(AUDIT_NOTE)).toBeVisible();
  });
  it('refuses a result it cannot validate', () => {
    render_card({ status: 'completed', member_count: 'many' });
    expect(
      screen.getByText('The dataset result format is not supported.'),
    ).toBeVisible();
    expect(screen.queryByText(AUDIT_NOTE)).not.toBeInTheDocument();
  });
  it('renders nothing for other operations or unfinished downloads', () => {
    const { container, rerender } = render(
      <DatasetSummaryCard
        job={{ ...saved_job, operation: 'train', result: fetch_summary }}
      />,
    );
    expect(container).toBeEmptyDOMElement();
    rerender(
      <DatasetSummaryCard
        job={{ ...saved_job, operation: 'fetch_dataset', result: null }}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
