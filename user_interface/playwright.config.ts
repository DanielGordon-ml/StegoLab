import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './browser_tests',
  timeout: 90000,
  expect: { timeout: 15000 },
  // One worker: the workspace spec restarts the backend, which would break any
  // encode or decode job running in a parallel worker.
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: process.env.STEGOLAB_BASE_URL ?? 'http://127.0.0.1:8080',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
