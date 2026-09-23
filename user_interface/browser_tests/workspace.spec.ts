import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

/** Exercise the real server; CI also restarts the backend to check disk persistence. */
test('workspace tabs and saved configuration survive reload and optional backend restart', async ({
  page,
  request,
}) => {
  const browser_errors: string[] = [];
  page.on('pageerror', (error) => browser_errors.push(error.message));
  await page.goto('/');
  await expect(page.getByText('Backend connected')).toBeVisible();
  for (const label of ['Encode', 'Decode', 'Train']) {
    await page.getByRole('tab', { name: label, exact: true }).click();
    await expect(page.getByRole('tabpanel', { name: label })).toContainText(
      'not available yet.',
    );
  }
  await page.getByRole('tab', { name: /Config/ }).click();
  const frequency = page.getByRole('spinbutton', {
    name: 'Checkpoint frequency',
  });
  await expect(frequency).toBeVisible();
  const initial_minutes = await frequency.inputValue();
  const saved_minutes = initial_minutes === '7' ? '8' : '7';
  await frequency.fill(saved_minutes);
  await page.getByRole('button', { name: /Save settings/ }).click();
  await expect(page.getByText('Settings saved.')).toBeVisible();
  await expect(page.getByText('Unsaved changes')).toHaveCount(0);

  if (process.env.STEGOLAB_RESTART_BACKEND === '1') {
    execFileSync(
      'docker',
      [
        'compose',
        '-f',
        '../infrastructure/compose.yaml',
        'restart',
        'backend_service',
      ],
      {
        cwd: fileURLToPath(new URL('..', import.meta.url)),
        timeout: 45000,
        stdio: 'pipe',
      },
    );
    await expect
      .poll(
        async () => {
          try {
            return (
              await request.get('/api/v1/health', { timeout: 2000 })
            ).status();
          } catch {
            return 0;
          }
        },
        { timeout: 30000 },
      )
      .toBe(200);
  }
  await page.reload();
  await page.getByRole('tab', { name: /Config/ }).click();
  await expect(frequency).toHaveValue(saved_minutes);
  await page.getByRole('button', { name: 'Reset to defaults' }).click();
  await expect(page.getByText('Defaults restored and saved.')).toBeVisible();
  await expect(frequency).toHaveValue('5');
  await page.reload();
  await page.getByRole('tab', { name: /Config/ }).click();
  await expect(frequency).toHaveValue('5');

  const accessibility = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
    .analyze();
  expect(accessibility.violations).toEqual([]);
  await page.getByRole('tab', { name: /Config/ }).focus();
  await page.keyboard.press('Home');
  await expect(
    page.getByRole('tab', { name: 'Encode', exact: true }),
  ).toBeFocused();
  await page.keyboard.press('ArrowRight');
  await expect(
    page.getByRole('tab', { name: 'Decode', exact: true }),
  ).toBeFocused();
  expect(browser_errors).toEqual([]);
});
