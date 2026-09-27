import { merge_job_lists } from '../contracts/job_merge';
import { useState } from 'react';
import {
  useMutation,
  useQueryClient,
  type UseMutationResult,
} from '@tanstack/react-query';
import { start_workflow } from '../contracts/workflow_service';
import { RequestFailure } from '../contracts/errors';
import type {
  JobList,
  JobSnapshot,
  WorkflowRequest,
} from '../contracts/workflows';

/** Everything a form needs to submit, lock, and retry one frozen request. */
export type WorkflowHandle<Request> = UseMutationResult<
  JobSnapshot,
  Error,
  Request
> & {
  submit: (request: Request) => void;
  locked: boolean;
  uncertain: boolean;
  retry: () => void;
};

/** Freeze a submitted operation so an uncertain result can be retried safely. */
export function useWorkflow(
  on_success?: (job: JobSnapshot) => void,
): WorkflowHandle<WorkflowRequest>;
export function useWorkflow<Request>(
  on_success: ((job: JobSnapshot) => void) | undefined,
  start: (request: Request) => Promise<JobSnapshot>,
): WorkflowHandle<Request>;
export function useWorkflow<Request>(
  on_success?: (job: JobSnapshot) => void,
  start?: (request: Request) => Promise<JobSnapshot>,
): WorkflowHandle<Request> {
  const start_request = (start ?? start_workflow) as (
    request: Request,
  ) => Promise<JobSnapshot>;
  const client = useQueryClient();
  const [attempt, set_attempt] = useState<Request | null>(null);
  const mutation = useMutation({
    mutationFn: start_request,
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
  function submit(request: Request) {
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
