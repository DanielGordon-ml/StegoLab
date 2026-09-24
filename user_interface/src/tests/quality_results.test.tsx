import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { expect, it } from 'vitest';
import { QualityReview } from '../components/quality_review';
import { EvaluationResultCard } from '../components/evaluation_result';
import { JobMonitor } from '../components/job_monitor';
import { saved_job } from './workflow_fixtures';

it('shows separate image quality counts and preserves unknown legacy counts', () => {
  const measured = {
    exact_recovery_rate: 0,
    trial_count: 3,
    psnr: 26,
    psnr_sample_count: 2,
    ssim: 0.8,
    ssim_sample_count: 1,
  };
  const { rerender } = render(
    <QualityReview checkpoints={[]} run_metrics={measured} />,
  );
  expect(screen.getByText('PSNR samples: 2')).toBeVisible();
  expect(screen.getByText('SSIM samples: 1')).toBeVisible();
  expect(screen.getByText('3 measured trials')).toBeVisible();
  rerender(
    <QualityReview
      checkpoints={[]}
      run_metrics={{
        ...measured,
        psnr_sample_count: null,
        ssim_sample_count: null,
      }}
    />,
  );
  expect(screen.getByText('PSNR samples: Not recorded')).toBeVisible();
  expect(screen.getByText('SSIM samples: Not recorded')).toBeVisible();
});

it('counts only each recorded image metric in a saved evaluation result', () => {
  const trial = {
    source_identity: 'public',
    width: 1024,
    height: 1024,
    fixture_kind: 'maximum',
    recovered: false,
    raw_bit_error_rate: null,
    peak_signal_to_noise_ratio: null,
    structural_similarity: null,
    identical_pixels: false,
    clipped_fraction: null,
    elapsed_seconds: 1,
    error_code: null,
  };
  const report = {
    schema_version: 1,
    kind: 'proof',
    compatibility_identifier: 'model_identity',
    dataset_revision: 'revision_1',
    completed: true,
    expected_bit_cases: 0,
    bit_cases_completed: 0,
    raw_bit_error_rate: null,
    expected_png_trials: 3,
    png_trials: [
      { ...trial, peak_signal_to_noise_ratio: 28, structural_similarity: 0.8 },
      { ...trial, peak_signal_to_noise_ratio: 24 },
      { ...trial, error_code: 'recovery_failed' },
    ],
    exact_recovery_count: 0,
    exact_recovery_rate: 0,
    median_peak_signal_to_noise_ratio: 26,
    median_structural_similarity: 0.8,
    mean_clipped_fraction: null,
    elapsed_seconds: 3,
    peak_process_memory_bytes: 100,
    learning_gate_passed: false,
    stopped_reason: 'complete',
  };
  render(
    <EvaluationResultCard
      job={{
        ...saved_job,
        operation: 'evaluate',
        status: 'completed',
        result: report,
      }}
    />,
  );
  expect(screen.getByText('PSNR samples: 2')).toBeVisible();
  expect(screen.getByText('SSIM samples: 1')).toBeVisible();
  expect(screen.getByText(/3 \/ 3 PNG trials/)).toBeVisible();
});

it('marks retained snapshots as last known and disables actions when status cannot be confirmed', () => {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <JobMonitor job={saved_job} stream_connected={true} confirmed={false} />
    </QueryClientProvider>,
  );
  expect(screen.getByText('Last known: queued')).toBeVisible();
  expect(
    screen.getByText(
      'Job status unavailable. Showing the last saved snapshot.',
    ),
  ).toBeVisible();
  expect(screen.getByRole('button', { name: 'Cancel job' })).toBeDisabled();
});
