import { useRef, useState, type KeyboardEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { read_capabilities, read_health } from './contracts/service';
import { ConfigurationPanel } from './components/configuration_panel';
import { ErrorNotice } from './components/error_notice';
import { Icon } from './components/icons';
import { TrainingPanel } from './components/training_panel';
import { InferencePanel } from './components/inference_panel';
import { is_active_job, useWorkspace } from './hooks/use_workspace';

const tabs = ['train', 'encode', 'decode', 'config'] as const;
type Tab = (typeof tabs)[number];
const tab_labels = {
  encode: 'Encode',
  decode: 'Decode',
  train: 'Train',
  config: 'Config',
};

/** Keep background jobs and connection state alive across accessible tabs. */
export function Application() {
  const [active_tab, set_active_tab] = useState<Tab>('train');
  const tab_buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const health = useQuery({
    queryKey: ['health'],
    queryFn: read_health,
    refetchInterval: 30000,
  });
  const capabilities = useQuery({
    queryKey: ['capabilities'],
    queryFn: read_capabilities,
    refetchInterval: 30000,
  });
  const state = useWorkspace();
  const disconnected = health.isError || capabilities.isError;
  const connected = health.isSuccess && capabilities.isSuccess && !disconnected;
  const active_job = state.jobs.data?.items.find(is_active_job);

  /** Follow the horizontal tab pattern, including Home and End. */
  function navigate_tabs(
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) {
    let next_index: number;
    if (event.key === 'ArrowRight') next_index = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft')
      next_index = (index + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next_index = 0;
    else if (event.key === 'End') next_index = tabs.length - 1;
    else return;
    event.preventDefault();
    set_active_tab(tabs[next_index]);
    tab_buttons.current[next_index]?.focus();
  }
  return (
    <div className="application_shell">
      <a href="#workspace" className="skip_link">
        Skip to workspace
      </a>
      <header className="topbar">
        <a className="brand" href="./" aria-label="StegoLab home">
          <span className="brand_mark">
            <Icon name="brand" />
          </span>
          <span>
            Stego<span className="brand_light">Lab</span>
          </span>
        </a>
        <span className="workspace_label">LOCAL WORKSPACE</span>
        <div className="header_resources">
          <span>
            {connected
              ? capabilities.data.available_devices.join(', ').toUpperCase()
              : 'Compute unknown'}
          </span>
          <span>
            {connected
              ? `${capabilities.data.available_models.length} installed models · experimental`
              : 'Models unknown'}
          </span>
        </div>
        <div
          className={`connection_status ${disconnected ? 'disconnected' : ''}`}
          role="status"
        >
          <span className={`status_dot ${connected ? 'ready' : ''}`} />
          {disconnected
            ? 'Disconnected'
            : connected
              ? 'Backend connected'
              : 'Connecting…'}
        </div>
      </header>
      <main id="workspace" tabIndex={-1}>
        <div className="workspace_intro">
          <div>
            <span className="eyebrow">IMAGE STEGANOGRAPHY</span>
            <p>Your local image lab.</p>
          </div>
          <span className="release_badge">
            TRAINING WORKSPACE <span>01</span>
          </span>
        </div>
        <div className="navigation_row">
          <div role="tablist" aria-label="Workspace tabs" className="tablist">
            {tabs.map((tab, index) => (
              <button
                key={tab}
                id={`tab_${tab}`}
                type="button"
                role="tab"
                aria-selected={active_tab === tab}
                aria-controls={`panel_${tab}`}
                tabIndex={active_tab === tab ? 0 : -1}
                ref={(element) => {
                  tab_buttons.current[index] = element;
                }}
                onClick={() => set_active_tab(tab)}
                onKeyDown={(event) => navigate_tabs(event, index)}
              >
                <Icon name={tab} />
                {tab_labels[tab]}
              </button>
            ))}
          </div>
          <button
            className={`active_job_indicator ${active_job && !state.jobs.isError ? 'has_active_job' : ''}`}
            onClick={() => set_active_tab('train')}
            aria-label={
              state.jobs.isError
                ? 'Job status unavailable'
                : active_job
                  ? `View active job: ${active_job.experiment_identifier ?? active_job.operation}`
                  : 'View training workspace'
            }
          >
            <span
              className={`status_dot ${active_job && !state.jobs.isError ? 'ready' : ''}`}
            />
            {state.jobs.isError
              ? active_job
                ? `${active_job.experiment_identifier || active_job.operation} · last known ${active_job.status}`
                : 'Job status unavailable'
              : active_job
                ? `${active_job.experiment_identifier || active_job.operation} · ${active_job.status}`
                : 'No active jobs'}
          </button>
        </div>
        {disconnected && (
          <div className="connection_banner">
            <ErrorNotice error={health.error ?? capabilities.error} />
            <button
              className="button secondary"
              onClick={() => {
                void health.refetch();
                void capabilities.refetch();
                void state.jobs.refetch();
                void state.workspace.refetch();
              }}
            >
              Reconnect
            </button>
          </div>
        )}
        {tabs.map((tab) => (
          <section
            key={tab}
            id={`panel_${tab}`}
            role="tabpanel"
            aria-labelledby={`tab_${tab}`}
            tabIndex={0}
            hidden={active_tab !== tab}
          >
            {tab === 'train' ? (
              <TrainingPanel state={state} />
            ) : tab === 'config' ? (
              <ConfigurationPanel />
            ) : active_tab === tab ? (
              <InferencePanel mode={tab} capabilities={capabilities.data} />
            ) : null}
          </section>
        ))}
        <section className="resource_strip" aria-label="Workspace resources">
          <div>
            <span className="resource_icon">◎</span>
            <div>
              <span className="resource_label">COMPUTE</span>
              <strong>
                {connected
                  ? capabilities.data.available_devices.join(', ').toUpperCase()
                  : 'Not confirmed'}
              </strong>
            </div>
          </div>
          <div>
            <span className="resource_icon">◇</span>
            <div>
              <span className="resource_label">INSTALLED MODELS</span>
              <strong>
                {connected
                  ? `${capabilities.data.available_models.length} models`
                  : 'Not confirmed'}
              </strong>
            </div>
          </div>
          <div>
            <span className="resource_icon">↗</span>
            <div>
              <span className="resource_label">PAYLOAD CAPACITY</span>
              <strong>
                {connected && capabilities.data.maximum_payload_bytes > 0
                  ? `${capabilities.data.maximum_payload_bytes} bytes available`
                  : 'Not available'}
              </strong>
            </div>
          </div>
          <span className="resource_note">
            Private workspace. Measured results.
          </span>
        </section>
      </main>
      <footer>
        <span>
          StegoLab <span className="footer_dot">/</span>{' '}
          {health.data?.application_version ?? 'Local workspace'}
        </span>
        <span>Built for a closer look.</span>
      </footer>
    </div>
  );
}
