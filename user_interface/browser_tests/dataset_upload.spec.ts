import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import type { JobSnapshot } from '../src/contracts/workflows';
import { empty_workspace, fetch_job } from '../src/tests/workflow_fixtures';
import { png } from './png_fixture';
import { zip } from './zip_fixture';

const WCAG_TAGS = ['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'];
const AUDIT_NOTE = 'A near-duplicate audit has not been done.';
const CANCELLING_TEXT =
  'Cancelling and cleaning up. Waiting for the backend to confirm.';

/** Every field of the Capabilities contract, with all four dataset sources on. */
const all_sources_capabilities = {
  application_version: '0.1.0',
  available_devices: ['cpu'],
  available_models: [],
  available_profiles: [],
  encoding_available: false,
  decoding_available: false,
  training_available: true,
  experimental_models_only: true,
  maximum_payload_bytes: 0,
  minimum_image_side: 0,
  maximum_image_side: 0,
  maximum_upload_bytes: 16777216,
  maximum_dataset_upload_bytes: 2147483648,
  dataset_upload_chunk_bytes: 16777216,
  hugging_face_token_configured: false,
  dataset_source_kinds: [
    'server_folder',
    'upload',
    'hugging_face',
    'https_archive',
  ],
};

/** Disk use answered for the Config tab, which stays mounted behind Train. */
const dataset_storage = {
  cache_bytes: 0,
  cache_entries: 0,
  cache_unused_bytes: 0,
  cache_unused_entries: 0,
  raw_source_bytes: 0,
  raw_source_folders: 0,
  prepared_bytes: 0,
  prepared_revisions: 0,
  free_disk_bytes: 40000000000,
  minimum_free_bytes: 5000000000,
  active_fetch_jobs: 1,
  cleanup_available: false,
};

/** The same download after the backend accepted a cancel it has not finished. */
const cancelling_job: JobSnapshot = {
  ...fetch_job,
  phase: 'cancelling',
  requested_action: 'cancel',
  available_actions: ['cancel'],
  updated_at: '2026-09-24T10:01:00Z',
  latest_event_identifier: 2,
};

/** The same download once the backend reports it as paused. */
const paused_job: JobSnapshot = {
  ...fetch_job,
  status: 'paused',
  phase: 'paused',
  requested_action: null,
  available_actions: ['resume', 'cancel'],
  updated_at: '2026-09-24T10:02:00Z',
  latest_event_identifier: 3,
};

/** Collect uncaught browser errors so a test can assert there were none. */
function watch_errors(page: Page) {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

/** Run the accessibility audit and the overflow check at desktop and phone widths. */
async function check_layouts(page: Page, label: string) {
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 900 });
    const accessibility = await new AxeBuilder({ page })
      .withTags(WCAG_TAGS)
      .analyze();
    expect(accessibility.violations, `${label} at ${width}px`).toEqual([]);
    const fits = await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    );
    expect(fits, `${label} must not overflow the page at ${width}px`).toBe(
      true,
    );
  }
  await page.setViewportSize({ width: 1280, height: 900 });
}

/** Send a two-image archive through the real backend and read what it saved. */
test(
  'browser archive upload becomes a raw folder and a prepared dataset',
  { tag: '@backend' },
  async ({ page, request }) => {
    test.setTimeout(300000);
    const browser_errors = watch_errors(page);
    const stamp = Date.now().toString(36);
    const source_name = `browser_upload_${stamp}`;
    const dataset_name = `browser_prepared_${stamp}`;
    const archive = zip([
      { name: 'first.png', data: png(256, 256) },
      { name: 'second.png', data: png(256, 256, 30) },
    ]);
    await page.goto('/');
    await expect(page.getByText('Backend connected')).toBeVisible();
    await page.getByText('Prepare a dataset', { exact: true }).click();
    await page.getByRole('radio', { name: 'Upload an archive' }).check();
    await page.getByLabel('Archive file').setInputFiles({
      name: 'browser_upload.zip',
      mimeType: 'application/zip',
      buffer: archive,
    });
    await expect(page.getByText(/Upload complete/).first()).toBeVisible({
      timeout: 60000,
    });
    await expect(
      page.getByText(/Files after extraction: about 2\b/),
    ).toBeVisible({ timeout: 60000 });
    await page.getByLabel('Save raw files as').fill(source_name);
    await page.getByLabel('Prepared dataset name').fill(dataset_name);
    // A renamed raw folder needs a fresh check before the fetch is allowed.
    await page.getByRole('button', { name: 'Check the archive again' }).click();
    await expect(
      page.getByText(`data/${source_name} will be created`),
    ).toBeVisible({ timeout: 60000 });
    await check_layouts(page, 'Upload form');
    await page.getByRole('button', { name: 'Fetch and prepare' }).click();
    const monitor = page.getByRole('region', {
      name: 'Dataset download',
      exact: true,
    });
    await expect(monitor.locator('.status_pill')).toHaveText('completed', {
      timeout: 200000,
    });
    const summary = page.getByRole('region', {
      name: 'Dataset download result',
    });
    await expect(summary).toContainText('Checked 2 image files');
    await expect(summary).toContainText(`data/${source_name}`);
    await expect(summary).toContainText(AUDIT_NOTE);
    const workspace = await (await request.get('/api/v1/workspace')).json();
    expect(JSON.stringify(workspace.sources)).toContain(source_name);
    expect(JSON.stringify(workspace.datasets)).toContain(dataset_name);
    await check_layouts(page, 'Download summary');
    expect(browser_errors).toEqual([]);
  },
);

/** Drive the monitor from stubbed snapshots: the backend decides which actions exist. */
test(
  'cancel and pause controls follow the backend',
  { tag: '@stubbed' },
  async ({ page }) => {
    const browser_errors = watch_errors(page);
    let current: JobSnapshot = fetch_job;
    const unstubbed: string[] = [];
    const actions: string[] = [];
    await page.route('**/api/v1/**', (route) => {
      unstubbed.push(route.request().url());
      return route.fulfill({ status: 404, json: {} });
    });
    await page.route('**/api/v1/health', (route) =>
      route.fulfill({
        json: { status: 'ready', application_version: '0.1.0' },
      }),
    );
    await page.route('**/api/v1/capabilities', (route) =>
      route.fulfill({ json: all_sources_capabilities }),
    );
    await page.route('**/api/v1/configuration', (route) =>
      route.fulfill({
        json: { schema_version: 1, checkpoint_interval_seconds: 300 },
      }),
    );
    await page.route('**/api/v1/workspace', (route) =>
      route.fulfill({ json: empty_workspace }),
    );
    await page.route('**/api/v1/datasets/storage', (route) =>
      route.fulfill({ json: dataset_storage }),
    );
    await page.route('**/api/v1/jobs', (route) =>
      route.fulfill({ json: { items: [current] } }),
    );
    await page.route('**/api/v1/events', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: `retry: 1000\nevent: reset\ndata: ${JSON.stringify({ items: [current] })}\n\n`,
      }),
    );
    await page.route('**/api/v1/jobs/*/actions', (route) => {
      actions.push(String(route.request().postDataJSON().action));
      current = cancelling_job;
      return route.fulfill({ json: current });
    });
    await page.goto('/');
    await expect(page.getByText('Backend connected')).toBeVisible();
    const monitor = page.getByRole('region', {
      name: 'Dataset download',
      exact: true,
    });
    await expect(monitor.getByText('12.0 of 100.0 MB')).toBeVisible();
    await expect(monitor.getByText('Downloading')).toBeVisible();
    await expect(
      monitor.getByRole('button', { name: 'Pause download' }),
    ).toBeEnabled();
    await monitor.getByRole('button', { name: 'Cancel job' }).click();
    await expect(
      monitor.getByRole('status').filter({ hasText: CANCELLING_TEXT }),
    ).toBeVisible();
    await expect(
      monitor.getByText('Cancelling', { exact: true }),
    ).toBeVisible();
    await expect(monitor.getByRole('button')).toHaveCount(0);
    expect(actions).toEqual(['cancel']);

    current = paused_job;
    await expect(
      monitor.getByRole('button', { name: 'Resume download' }),
    ).toBeEnabled();
    await expect(
      monitor.getByRole('button', { name: 'Cancel job' }),
    ).toBeEnabled();
    await expect(
      monitor.getByRole('button', { name: 'Pause download' }),
    ).toHaveCount(0);
    await expect(monitor.getByText('12.0 of 100.0 MB')).toBeVisible();
    expect(unstubbed).toEqual([]);
    expect(browser_errors).toEqual([]);
  },
);
