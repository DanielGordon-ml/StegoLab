import {
  test,
  expect,
  type APIRequestContext,
  type Page,
} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';

const PACKAGE =
  process.env.STEGOLAB_MODEL_PACKAGE ?? 'fixture_not_a_neural_model';
const MESSAGE = 'Sunny garden — שלום 🌻 browser round trip';
const PASSWORD = 'browser-test-password';
const RECOVERY_MESSAGE =
  'No valid hidden message could be recovered. Check the password, model, and image.';

/** Compute the checksum every PNG chunk carries. */
function crc32(bytes: Uint8Array) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1)
      crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/** Wrap one chunk with its length, name and checksum. */
function chunk(name: string, data: Uint8Array) {
  const header = Buffer.alloc(8);
  header.writeUInt32BE(data.length, 0);
  header.write(name, 4, 'ascii');
  const body = Buffer.concat([Buffer.from(name, 'ascii'), Buffer.from(data)]);
  const footer = Buffer.alloc(4);
  footer.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([header, Buffer.from(data), footer]);
}

/** Build a smooth RGB PNG in memory so the test needs no image files. */
function png(width: number, height: number) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr.set([8, 2, 0, 0, 0], 8);
  const rows = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y += 1) {
    const offset = y * (width * 3 + 1);
    rows[offset] = 0;
    for (let x = 0; x < width; x += 1) {
      const pixel = offset + 1 + x * 3;
      rows[pixel] = (40 + (x * 200) / width) | 0;
      rows[pixel + 1] = (40 + (y * 200) / height) | 0;
      rows[pixel + 2] = (120 + ((x + y) * 100) / (width + height)) | 0;
    }
  }
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(rows)),
    chunk('IEND', new Uint8Array()),
  ]);
}

/** Read the dimensions recorded in a PNG header. */
function dimensions(bytes: Buffer) {
  return [bytes.readUInt32BE(16), bytes.readUInt32BE(20)];
}

/** List the decode jobs the backend currently knows about. */
async function decode_jobs(request: APIRequestContext) {
  const jobs = await (await request.get('/api/v1/jobs')).json();
  return jobs.items.filter(
    (job: { operation: string }) => job.operation === 'decode',
  ) as { job_identifier: string; status: string }[];
}

/** Start one decode and wait until a new job for it reaches a final state. */
async function decode(
  page: Page,
  request: APIRequestContext,
  file: Buffer,
  password: string,
) {
  const before = new Set(
    (await decode_jobs(request)).map((job) => job.job_identifier),
  );
  await page.getByRole('tab', { name: 'Decode', exact: true }).click();
  await page.getByLabel('Encoded PNG').setInputFiles({
    name: 'encoded.png',
    mimeType: 'image/png',
    buffer: file,
  });
  await expect(page.getByText(/512×512 px/)).toBeVisible();
  await page.getByLabel('Matching decoder model').selectOption(PACKAGE);
  await page.getByLabel('Password').fill(password);
  await page.getByRole('button', { name: 'Authenticate and decode' }).click();
  let outcome = { job_identifier: '', status: 'pending' };
  await expect
    .poll(
      async () => {
        const fresh = (await decode_jobs(request)).filter(
          (job) => !before.has(job.job_identifier),
        );
        if (fresh.length !== 1) return 'pending';
        outcome = fresh[0];
        return ['queued', 'running'].includes(outcome.status)
          ? 'pending'
          : outcome.status;
      },
      { timeout: 200000 },
    )
    .not.toBe('pending');
  return outcome;
}

test.describe.configure({ timeout: 240000 });

/** Exercise the real stack with the labelled fixture pair or a named package. */
test('encode a message, download the verified PNG, decode it, and clear the text', async ({
  page,
  context,
  request,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const browser_errors: string[] = [];
  page.on('pageerror', (error) => browser_errors.push(error.message));
  await page.goto('/');
  await expect(page.getByText('Backend connected')).toBeVisible();
  const exports = page.getByRole('listitem').filter({ hasText: PACKAGE });
  await expect(exports.first()).toBeVisible();
  const install = exports.getByRole('button', {
    name: 'Install as experimental model',
  });
  const installed = exports.getByText('Installed');
  await expect(install.first().or(installed)).toBeVisible();
  if ((await installed.count()) === 0) await install.first().click();
  await expect(installed).toBeVisible();

  await page.getByRole('tab', { name: 'Encode', exact: true }).click();
  await expect(page.getByText('Experimental model.')).toBeVisible();
  const cover = png(512, 512);
  await page
    .getByLabel('Cover image')
    .setInputFiles({ name: 'cover.png', mimeType: 'image/png', buffer: cover });
  await expect(page.getByText(/512×512 px/)).toBeVisible();
  await page.getByLabel('Encoder model').selectOption(PACKAGE);
  await page.getByLabel('Message').fill(MESSAGE);
  await expect(page.getByText(/of 256 bytes used/)).toBeVisible();
  await page.getByLabel('Password').fill(PASSWORD);
  const encode_jobs = async () =>
    (await (await request.get('/api/v1/jobs')).json()).items.filter(
      (job: { operation: string }) => job.operation === 'encode',
    ).length;
  const encode_count = await encode_jobs();
  await page.getByRole('button', { name: 'Encode and verify' }).dblclick();
  await expect(page.getByLabel('Password')).toHaveValue('');
  await expect(
    page.getByText('The saved PNG passed verification.'),
  ).toBeVisible({ timeout: 200000 });
  expect(await encode_jobs()).toBe(encode_count + 1);
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('link', { name: 'Download verified PNG' }).click(),
  ]);
  await expect(page.getByText('Download started.')).toBeVisible();
  const encoded = readFileSync((await download.path()) as string);
  expect(encoded.subarray(0, 8)).toEqual(
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
  );
  expect(dimensions(encoded)).toEqual([512, 512]);
  expect(download.suggestedFilename()).toMatch(
    /^stegolab-encoded-[0-9a-f]{8}\.png$/,
  );

  const first = await decode(page, request, encoded, PASSWORD);
  expect(first.status).toBe('completed');
  const recovered = page.getByLabel('Recovered message');
  await expect(recovered).toHaveValue(MESSAGE);
  await page.getByRole('button', { name: 'Copy message' }).click();
  await expect(page.getByText('Copied to the clipboard.')).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
    MESSAGE,
  );
  await page.getByRole('button', { name: 'Clear message' }).click();
  await expect(recovered).toHaveValue('');
  await expect
    .poll(async () =>
      (
        await request.get(`/api/v1/jobs/${first.job_identifier}/decoded_text`)
      ).status(),
    )
    .toBe(404);

  const wrong = await decode(page, request, encoded, 'wrong-password');
  expect(wrong.status).toBe('failed');
  await expect(page.getByText(RECOVERY_MESSAGE)).toBeVisible();
  await page.getByRole('tab', { name: 'Encode', exact: true }).click();
  const plain = await decode(page, request, cover, PASSWORD);
  expect(plain.status).toBe('failed');
  await expect(page.getByText(RECOVERY_MESSAGE)).toBeVisible();

  for (const label of ['Encode', 'Decode']) {
    await page.getByRole('tab', { name: label, exact: true }).click();
    const accessibility = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
      .analyze();
    expect(accessibility.violations, label).toEqual([]);
  }
  expect(browser_errors).toEqual([]);
});

/** Show the installation hint and keep controls locked when no model exists. */
test('without an installed model both tabs explain installation', async ({
  page,
}) => {
  const zero_models = {
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
  };
  await page.route('**/api/v1/capabilities', (route) =>
    route.fulfill({ json: zero_models }),
  );
  await page.route('**/api/v1/models', (route) =>
    route.fulfill({ json: { items: [] } }),
  );
  await page.goto('/');
  for (const label of ['Encode', 'Decode']) {
    await page.getByRole('tab', { name: label, exact: true }).click();
    const panel = page.getByRole('tabpanel', { name: label });
    await expect(
      panel.getByText('No experimental model is installed.'),
    ).toBeVisible();
    await expect(panel.getByRole('combobox')).toBeDisabled();
    await expect(
      panel.getByRole('button', { name: /verify|decode/ }),
    ).toBeDisabled();
    const accessibility = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
      .analyze();
    expect(accessibility.violations, label).toEqual([]);
  }
});
