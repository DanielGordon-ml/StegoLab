import { useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { read_models } from '../contracts/inference_service';
import { ErrorNotice } from './error_notice';

/** Choose one explicitly installed experimental model, never an unverified export. */
export function ModelChoice({
  id,
  label,
  value,
  disabled,
  on_change,
}: {
  id: string;
  label: string;
  value: string;
  disabled?: boolean;
  on_change: (model_identifier: string) => void;
}) {
  const models = useQuery({
    queryKey: ['models'],
    queryFn: read_models,
    refetchInterval: 30000,
  });
  const items = useMemo(() => models.data?.items ?? [], [models.data]);
  const known = items.some((item) => item.model_identifier === value);
  useEffect(() => {
    if (!known && items.length > 0) on_change(items[0].model_identifier);
    if (known || items.length > 0 || !value) return;
    on_change('');
  }, [items, known, value, on_change]);
  return (
    <>
      <label htmlFor={id}>{label}</label>
      <select
        id={id}
        value={known ? value : ''}
        disabled={disabled || items.length === 0}
        onChange={(event) => on_change(event.target.value)}
      >
        {items.length === 0 ? (
          <option value="">No experimental model installed</option>
        ) : (
          items.map((item) => (
            <option key={item.model_identifier} value={item.model_identifier}>
              {item.model_identifier} · experimental · {item.minimum_side}–
              {item.maximum_side} px
            </option>
          ))
        )}
      </select>
      <ErrorNotice error={models.error} />
    </>
  );
}
