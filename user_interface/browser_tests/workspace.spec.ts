import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { empty_workspace } from '../src/tests/workflow_fixtures';

/** Exercise the real server; CI also restarts the backend to check disk persistence. */
test('workspace tabs and saved configuration survive reload and optional backend restart', async ({
  page,
  request,
}) => {
  const browser_errors: string[] = [];
  page.on('pageerror', (error) => browser_errors.push(error.message));
  await page.goto('/');
  await expect(page.getByText('Backend connected')).toBeVisible();
  await expect(
    page.getByRole('tab', { name: 'Train', exact: true }),
  ).toHaveAttribute('aria-selected', 'true');
  await expect(
    page.getByRole('button', { name: 'New training run' }),
  ).toBeVisible();
  for (const label of ['Encode', 'Decode']) {
    await page.getByRole('tab', { name: label, exact: true }).click();
    await expect(page.getByRole('tabpanel', { name: label })).toContainText(
      /model/i,
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
    page.getByRole('tab', { name: 'Train', exact: true }),
  ).toBeFocused();
  await page.keyboard.press('ArrowRight');
  await expect(
    page.getByRole('tab', { name: 'Encode', exact: true }),
  ).toBeFocused();
  expect(browser_errors).toEqual([]);
});

/** Verify every new panel under desktop and narrow keyboard-friendly layouts. */
test('all workspace panels are accessible and fit a narrow screen', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.getByText('Backend connected')).toBeVisible();
  await expect(
    page.getByRole('button', { name: 'New training run' }),
  ).toBeEnabled();
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 900 });
    for (const label of ['Train', 'Encode', 'Decode', 'Config']) {
      await page.getByRole('tab', { name: label, exact: true }).click();
      await expect(page.getByRole('tabpanel', { name: label })).toBeVisible();
      const accessibility = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
        .analyze();
      expect(accessibility.violations, `${label} at ${width}px`).toEqual([]);
      const fits = await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      );
      expect(fits, `${label} must not overflow the page at ${width}px`).toBe(
        true,
      );
    }
  }
});

/** Check a fresh install without touching live datasets, jobs, or saved settings. */
test('empty training workspace stays accessible on desktop and narrow screens', async ({
  page,
}) => {
  await page.route('**/api/v1/workspace', (route) =>
    route.fulfill({ json: empty_workspace }),
  );
  await page.route('**/api/v1/jobs', (route) =>
    route.fulfill({ json: { items: [] } }),
  );
  await page.route('**/api/v1/events', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'event: reset\ndata: {"items":[]}\n\n',
    }),
  );
  await page.goto('/');
  await expect(
    page.getByRole('tab', { name: 'Train', exact: true }),
  ).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText('No linked evaluation yet')).toHaveCount(2);
  await expect(
    page.getByText(
      'No runs yet. Choose a prepared dataset to start your first experiment.',
    ),
  ).toBeVisible();
  // Axe cannot resolve contrast over gradients. Check each rendered stop directly.
  const contrast_ratios = await page
    .locator('.chart_empty')
    .evaluateAll((elements) => {
      /** Convert a rendered RGB color to WCAG relative luminance. */
      function luminance(color: string) {
        const channels = color
          .match(/[\d.]+/g)!
          .slice(0, 3)
          .map(Number)
          .map((value) => {
            const channel = value / 255;
            return channel <= 0.04045
              ? channel / 12.92
              : ((channel + 0.055) / 1.055) ** 2.4;
          });
        return (
          channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
        );
      }
      return elements.flatMap((element) => {
        const style = getComputedStyle(element);
        const foreground = luminance(style.color);
        const backgrounds = style.backgroundImage.match(
          /rgba?\([\d.,\s]+\)/g,
        ) ?? [style.backgroundColor];
        return backgrounds.map((background) => {
          const lightness = luminance(background);
          return (
            (Math.max(foreground, lightness) + 0.05) /
            (Math.min(foreground, lightness) + 0.05)
          );
        });
      });
    });
  expect(contrast_ratios.length).toBeGreaterThan(0);
  for (const ratio of contrast_ratios)
    expect(ratio).toBeGreaterThanOrEqual(4.5);
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 900 });
    const accessibility = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
      .analyze();
    expect(accessibility.violations, `Empty Train at ${width}px`).toEqual([]);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
  }
});
