import { merge_job_lists } from '../contracts/job_merge';
import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { start_workflow } from '../contracts/workflow_service';
import { RequestFailure } from '../contracts/errors';
import type {
  JobList,
  JobSnapshot,
  WorkflowRequest,
} from '../contracts/workflows';

/** Freeze a submitted operation so an uncertain result can be retried safely. */
export function useWorkflow(on_success?: (job: JobSnapshot) => void) {
  const client = useQueryClient();
  const [attempt, set_attempt] = useState<WorkflowRequest | null>(null);
  const mutation = useMutation({
    mutationFn: start_workflow,
    onSuccess: (job) => {
      client.setQueryData<JobList>(['jobs'], (previous) =>
        merge_job_lists(previous, { items: [job] }),
      );
      void client.invalidateQueries({ queryKey: ['workspace'] });
      set_attempt(null);
      on_success?.(job);
    },
  });
  const uncertain =
    mutation.isError &&
    mutation.error instanceof RequestFailure &&
    mutation.error.uncertain;
  /** Store the request before sending; no automatic mutation retries. */
  function submit(request: WorkflowRequest) {
    set_attempt(request);
    mutation.mutate(request);
  }
  return {
    ...mutation,
    submit,
    locked: mutation.isPending || uncertain,
    uncertain,
    retry: () => {
      if (attempt) mutation.mutate(attempt);
    },
  };
}
