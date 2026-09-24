import type { JobList, JobSnapshot } from './workflows';

/** Prevent delayed reads or mutation responses from replacing newer event state. */
export function merge_job_lists(
  previous: JobList | undefined,
  incoming: JobList,
): JobList {
  const by_identifier = new Map(
    (previous?.items ?? []).map((job) => [job.job_identifier, job]),
  );
  for (const job of incoming.items) {
    const current = by_identifier.get(job.job_identifier);
    if (!current || is_newer(job, current))
      by_identifier.set(job.job_identifier, job);
  }
  return {
    items: [...by_identifier.values()].sort((first, second) =>
      first.created_at.localeCompare(second.created_at),
    ),
  };
}

/** Prefer durable event order, using timestamps only for older saved records. */
function is_newer(incoming: JobSnapshot, current: JobSnapshot) {
  const next_revision = incoming.latest_event_identifier ?? 0;
  const previous_revision = current.latest_event_identifier ?? 0;
  if (next_revision !== previous_revision)
    return next_revision > previous_revision;
  return incoming.updated_at > current.updated_at;
}
