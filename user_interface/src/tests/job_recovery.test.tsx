import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { PropsWithChildren } from 'react';
import { useWorkspace } from '../hooks/use_workspace';
import { useWorkflow } from '../hooks/use_workflow';
import { merge_job_lists } from '../contracts/job_merge';
import { validate_jobs } from '../contracts/workflow_service';
import { empty_workspace, saved_job } from './workflow_fixtures';

/** Emit exactly the public browser event contract without a live worker. */
class TestEventSource extends EventTarget {
  static current: TestEventSource;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  constructor() {
    super();
    TestEventSource.current = this;
  }
  close() {
    this.closed = true;
  }
}

/** Isolate query state while preserving real request and response validation. */
function query_context() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: PropsWithChildren) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }
  return { client, wrapper: Wrapper };
}
const completed = {
  ...saved_job,
  status: 'completed' as const,
  available_actions: [],
  latest_event_identifier: 4,
  updated_at: '2026-09-24T10:02:00Z',
};
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('durable job updates', () => {
  it('rejects unsupported validator input without breaking initial queries', () => {
    expect(validate_jobs(undefined)).toBe(false);
    expect(validate_jobs({ items: [] })).toBe(true);
    expect(
      merge_job_lists({ items: [completed] }, { items: [saved_job] }).items[0],
    ).toEqual(completed);
  });
  it('does not let a delayed jobs read replace a newer event snapshot', async () => {
    let complete_read: (response: Response) => void = () => {};
    vi.stubGlobal('EventSource', TestEventSource);
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) =>
        url.endsWith('/jobs')
          ? new Promise<Response>((resolve) => {
              complete_read = resolve;
            })
          : Response.json(empty_workspace),
      ),
    );
    const { client, wrapper } = query_context();
    const { result, unmount } = renderHook(useWorkspace, { wrapper });
    await act(async () => {
      TestEventSource.current.dispatchEvent(
        new MessageEvent('job', {
          data: JSON.stringify({
            event_identifier: 4,
            job_identifier: completed.job_identifier,
            status: completed.status,
            phase: completed.phase,
            created_at: completed.updated_at,
            snapshot: completed,
          }),
        }),
      );
    });
    expect(client.getQueryData(['jobs'])).toEqual({ items: [completed] });
    await act(async () => {
      complete_read(Response.json({ items: [saved_job] }));
    });
    await waitFor(() => expect(result.current.jobs.isFetching).toBe(false));
    expect(result.current.jobs.data).toEqual({ items: [completed] });
    unmount();
    expect(TestEventSource.current.closed).toBe(true);
  });
  it('does not let a delayed start response replace a completed snapshot', async () => {
    let finish: (response: Response) => void = () => {};
    vi.stubGlobal(
      'fetch',
      vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            finish = resolve;
          }),
      ),
    );
    const { client, wrapper } = query_context();
    const { result } = renderHook(() => useWorkflow(), { wrapper });
    act(() =>
      result.current.submit({
        operation: 'train',
        client_request_identifier: 'one_request',
        dataset_identifier: 'dataset_example',
      }),
    );
    await waitFor(() => expect(result.current.isPending).toBe(true));
    client.setQueryData(['jobs'], { items: [completed] });
    await act(async () => finish(Response.json(saved_job)));
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(client.getQueryData(['jobs'])).toEqual({ items: [completed] });
  });
  it('polls active snapshots every five seconds only while the stream is disconnected', async () => {
    vi.stubGlobal('EventSource', TestEventSource);
    const fetch = vi.fn(async (url: string) =>
      Response.json(
        url.endsWith('/jobs') ? { items: [saved_job] } : empty_workspace,
      ),
    );
    vi.stubGlobal('fetch', fetch);
    const { wrapper } = query_context();
    const { result, unmount } = renderHook(useWorkspace, { wrapper });
    await waitFor(() => expect(result.current.jobs.isSuccess).toBe(true));
    vi.useFakeTimers();
    await act(async () => {
      TestEventSource.current.onopen?.();
    });
    const connected_reads = fetch.mock.calls.filter(([url]) =>
      url.endsWith('/jobs'),
    ).length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(
      fetch.mock.calls.filter(([url]) => url.endsWith('/jobs')),
    ).toHaveLength(connected_reads);
    act(() => TestEventSource.current.onerror?.());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(
      fetch.mock.calls.filter(([url]) => url.endsWith('/jobs')),
    ).toHaveLength(connected_reads + 1);
    unmount();
  });
});
