import { merge_job_lists } from '../contracts/job_merge';
import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  read_jobs,
  read_workspace,
  validate_job_event,
  validate_jobs,
} from '../contracts/workflow_service';
import type { JobList, JobSnapshot } from '../contracts/workflows';

/** Identify jobs whose state can still change without another user request. */
export function is_active_job(job: JobSnapshot) {
  return job.status === 'queued' || job.status === 'running';
}

/** Own one event stream for the entire workspace, with a snapshot fallback. */
export function useWorkspace() {
  const client = useQueryClient();
  const [stream_connected, set_stream_connected] = useState(false);
  const jobs = useQuery({
    queryKey: ['jobs'],
    queryFn: read_jobs,
    structuralSharing: (previous, incoming) =>
      validate_jobs(previous) && validate_jobs(incoming)
        ? merge_job_lists(previous, incoming)
        : incoming,
    refetchInterval: (query) =>
      !stream_connected && query.state.data?.items.some(is_active_job)
        ? 5000
        : false,
  });
  const workspace = useQuery({
    queryKey: ['workspace'],
    queryFn: read_workspace,
    refetchInterval: jobs.data?.items.some(is_active_job) ? 10000 : 30000,
  });
  useEffect(() => {
    if (typeof EventSource === 'undefined') return;
    const stream = new EventSource('/api/v1/events');
    stream.onopen = () => {
      set_stream_connected(true);
      void client.invalidateQueries({ queryKey: ['jobs'] });
      void client.invalidateQueries({ queryKey: ['workspace'] });
    };
    stream.onerror = () => set_stream_connected(false);
    /** Parse untrusted events and merge only newer validated snapshots. */
    const accept_job = (event: MessageEvent<string>) => {
      try {
        const payload: unknown = JSON.parse(event.data);
        if (!validate_job_event(payload) || !payload.snapshot) return;
        const snapshot = payload.snapshot;
        client.setQueryData<JobList>(['jobs'], (previous) =>
          merge_job_lists(previous, { items: [snapshot] }),
        );
        if (!is_active_job(snapshot))
          void client.invalidateQueries({ queryKey: ['workspace'] });
      } catch {
        /* Ignore malformed event data; snapshot reads remain authoritative. */
      }
    };
    /** Replace snapshots after the server can no longer replay old events. */
    const accept_reset = (event: MessageEvent<string>) => {
      try {
        const payload: unknown = JSON.parse(event.data);
        if (validate_jobs(payload))
          client.setQueryData<JobList>(['jobs'], (previous) =>
            merge_job_lists(previous, payload),
          );
      } catch {
        /* Reconnection also requests a validated snapshot. */
      }
    };
    stream.addEventListener('job', accept_job);
    stream.addEventListener('reset', accept_reset);
    return () => stream.close();
  }, [client]);
  return { workspace, jobs, stream_connected };
}
