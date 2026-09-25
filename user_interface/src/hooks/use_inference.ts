import { useEffect, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { merge_job_lists } from '../contracts/job_merge';
import { start_inference } from '../contracts/inference_service';
import { RequestFailure } from '../contracts/errors';
import type {
  DecodingJobRequest,
  EncodingJobRequest,
} from '../contracts/inference';
import type { JobList, JobSnapshot } from '../contracts/workflows';

type InferenceRequest = EncodingJobRequest | DecodingJobRequest;

/** Tell a lost answer, which may be retried, from a definite refusal. */
function is_uncertain(error: unknown) {
  return error instanceof RequestFailure && error.uncertain;
}

/** Submit one encode or decode job, keeping its secrets in memory only until settled. */
export function useInferenceJob(on_success?: (job: JobSnapshot) => void) {
  const client = useQueryClient();
  const [attempt, set_attempt] = useState<InferenceRequest | null>(null);
  const [failure, set_failure] = useState<Error | null>(null);
  const mutation = useMutation({
    mutationFn: start_inference,
    gcTime: 0,
    onSuccess: (job) => {
      client.setQueryData<JobList>(['jobs'], (previous) =>
        merge_job_lists(previous, { items: [job] }),
      );
      set_attempt(null);
      on_success?.(job);
    },
    onError: (error) => {
      if (is_uncertain(error)) return;
      set_failure(error);
      set_attempt(null);
    },
  });
  const { isSuccess, isError, error, reset } = mutation;
  const uncertain = isError && is_uncertain(error);
  useEffect(() => {
    // Drop the settled request, and its secrets, from the shared mutation cache.
    if (isSuccess || (isError && !uncertain)) reset();
  }, [isSuccess, isError, uncertain, reset]);
  /** Freeze the request so an uncertain result can be retried with the same identity. */
  function submit(request: InferenceRequest) {
    set_failure(null);
    set_attempt(request);
    mutation.mutate(request);
  }
  /** Forget a frozen request, including its secrets, without sending it again. */
  function discard() {
    set_failure(null);
    set_attempt(null);
    mutation.reset();
  }
  return {
    isPending: mutation.isPending,
    error: mutation.error ?? failure,
    submit,
    discard,
    locked: mutation.isPending || uncertain,
    uncertain,
    retry: () => {
      if (attempt) mutation.mutate(attempt);
    },
  };
}
