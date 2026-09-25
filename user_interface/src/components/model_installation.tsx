import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { install_model, read_models } from '../contracts/inference_service';
import type { Workspace } from '../contracts/workspace';
import { ErrorNotice } from './error_notice';

/** Offer explicit installation of one exported pair as an experimental model. */
export function InstallModelButton({
  export_item,
}: {
  export_item: Workspace['exports'][number];
}) {
  const client = useQueryClient();
  const models = useQuery({ queryKey: ['models'], queryFn: read_models });
  const installed = models.data?.items.some(
    (item) => item.model_identifier === export_item.name,
  );
  const install = useMutation({
    mutationFn: install_model,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['models'] });
      void client.invalidateQueries({ queryKey: ['capabilities'] });
    },
  });
  if (installed)
    return <span className="status_pill completed">Installed</span>;
  if (models.isPending)
    return <span className="help_text">Checking installed models…</span>;
  return (
    <>
      <button
        className="button secondary"
        disabled={install.isPending || models.isPending}
        onClick={() =>
          install.mutate({
            client_request_identifier: crypto.randomUUID(),
            export_reference: export_item.identifier,
          })
        }
      >
        Install as experimental model
      </button>
      <ErrorNotice error={install.error} />
    </>
  );
}
