import { useEffect, useRef, useState } from 'react';
import {
  discard_decoded_text,
  read_decoded_text,
} from '../contracts/inference_service';
import type { DecodedText } from '../contracts/inference';
import type { JobSnapshot } from '../contracts/workflows';

interface TextState {
  identifier: string | null;
  text: DecodedText | null;
  error: Error | null;
}

/** Jobs whose text this page already forgot; never ask the backend for them again. */
export const discarded_texts = new Set<string>();

/** Ask the backend to forget one text and remember locally that it is gone. */
function forget(identifier: string) {
  discarded_texts.add(identifier);
  void discard_decoded_text(identifier).catch(() => {});
}

/** Hold recovered text in component memory only, and forget it on clear or leave. */
export function useDecodedText(job: JobSnapshot | undefined) {
  const ready = job?.status === 'completed' && job.operation === 'decode';
  const identifier = ready ? job.job_identifier : null;
  const [state, set_state] = useState<TextState>({
    identifier: null,
    text: null,
    error: null,
  });
  const [, force_render] = useState(0);
  const held = useRef<string | null>(null);
  const skip = !identifier || discarded_texts.has(identifier);
  useEffect(() => {
    if (skip) return;
    let stale = false;
    held.current = identifier;
    read_decoded_text(identifier)
      .then((text) => {
        if (stale) return;
        set_state({ identifier, text, error: null });
      })
      .catch((error: unknown) => {
        if (stale) return;
        held.current = null;
        set_state({
          identifier,
          text: null,
          error: error instanceof Error ? error : new Error('unavailable'),
        });
      });
    return () => {
      stale = true;
      // Leaving the tab or moving to another job forgets any text held so far.
      if (held.current === identifier) {
        held.current = null;
        forget(identifier);
      }
    };
  }, [identifier, skip]);
  const expires_at = state.text?.expires_at;
  useEffect(() => {
    if (!expires_at || !identifier) return;
    // Browsers treat delays above 2^31 - 1 ms as zero, so clamp far-off expiries.
    const delay = Math.min(
      Math.max(0, Date.parse(expires_at) - Date.now()),
      2_147_483_647,
    );
    const timer = setTimeout(() => {
      held.current = null;
      discarded_texts.add(identifier);
      force_render((count) => count + 1);
    }, delay);
    return () => clearTimeout(timer);
  }, [expires_at, identifier]);
  /** Forget the text locally and in the backend as soon as the reader is done. */
  async function clear() {
    held.current = null;
    set_state({ identifier, text: null, error: null });
    if (identifier) {
      discarded_texts.add(identifier);
      await discard_decoded_text(identifier).catch(() => {});
    }
  }
  const current = state.identifier === identifier && !skip;
  return {
    text: current ? state.text : null,
    error: current ? state.error : null,
    loading: !skip && !current,
    clear,
  };
}
