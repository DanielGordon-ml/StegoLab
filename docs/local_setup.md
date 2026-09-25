# Run StegoLab locally

Sprint 1 provides four tabs and saved settings. Sprint 2 adds message-protocol
checks and image preparation through the command line. Sprint 3 adds local
dataset preparation and offline integrity checks; see [datasets](datasets.md).
Sprint 4 adds experimental CPU training, checkpoint inspection, evaluation, and
independent model exports through the [command-line workflow](model_training.md).
The browser now connects the local CPU workflow through durable jobs; see the
[GUI training guide](gui_training.md). Encoding and decoding work with an
experimental model installed from the Train tab; dataset downloads remain
unavailable, and no model is approved for release.

## Start with Docker

- Install and start Docker Desktop with Compose. Allocate enough memory for your
  other work; the backend has a 10 GiB upper limit and the frontend has 256 MiB.
- Use port 8081 for this local setup so another application can keep port 8080.
  Create or edit `infrastructure/.env` once and add this line:

```dotenv
STEGOLAB_PORT=8081
```

- Git ignores this local file. Compose loads it beside `compose.yaml`, so the
  setting remains after you close the terminal. The shared default stays 8080
  when no override is set.
- An exported `STEGOLAB_PORT` in your shell overrides the file. Run
  `unset STEGOLAB_PORT` to use the saved value.
- Run these commands from the repository root:

```sh
sh infrastructure/setup_local_workspace.sh
docker compose -f infrastructure/compose.yaml up --build --wait
docker compose -f infrastructure/compose.yaml ps
```

Open [StegoLab](http://127.0.0.1:8081). Both containers must report healthy.
`--wait` runs the containers in the background and waits for them to be healthy.
The build uses native CPU images on Apple Silicon and Linux. No GPU or cloud
account is needed. Only the frontend publishes a port, bound to your computer.

If port 8081 is also busy, choose a free port in `infrastructure/.env`, run the
start command again, and use that port in the browser address. The port always
binds to localhost. Without a local override, open
[StegoLab on the default port](http://127.0.0.1:8080).

The fixed CPU training profile saves every five minutes. Config stores a separate
default for future configurable profiles; it does not change that fixed profile.
The Config tab starts with a five-minute checkpoint frequency. Save another
positive whole-minute value, restart the backend, then reload the page:

```sh
docker compose -f infrastructure/compose.yaml restart backend_service
```

The saved value must remain. Reset restores and saves five minutes. CPU and
qualified model availability are read-only.

## Storage and stopping

- The `stegolab_application_data` named volume stores settings and job metadata
  under `/data/state`, plus run logs under `/data/logs`.
- A new volume inherits the backend's non-root ownership during first startup.
  Do not replace it with a root-owned host directory.
- Training data, checkpoints, exported models, and the original experiment ledger
  stay in the repository's mounted folders. The setup helper creates these folders
  and configures write access for fresh Linux installations; see the
  [GUI training guide](gui_training.md) for existing installations.
- Docker output is limited to three 10 MiB files per service. Request access
  logs are disabled so request values do not enter proxy logs.
- Stop containers while keeping saved data:

```sh
docker compose -f infrastructure/compose.yaml down
```

Avoid `down --volumes` for your working installation: it deletes saved settings
and all other data in this volume. CI uses it only for disposable test data.

To start again after `down`, use the start command above. To restart existing
containers without changing their configuration:

```sh
docker compose -f infrastructure/compose.yaml restart
```

After changing the port or application code, use `up --build --wait` again.
`restart` does not apply those changes.

## Development checks

Install Python 3.12, uv 0.10.12, and Node.js 24. Dependency versions are locked.
Run from the repository root:

```sh
uv sync --frozen
npm --prefix user_interface ci
uv run ruff check .
uv run ruff format --check .
uv run mypy backend_service schemas
uv run pytest tests/backend
uv run stegolab verify_protocol
uv run python -m backend_service.export_contracts --check
uv run python scripts/check_file_lengths.py
npm --prefix user_interface run lint
npm --prefix user_interface test
npm --prefix user_interface run build
```

The file-length check includes generated source, but excludes lockfiles and
JSON schema data. To run the browser workflow, start Compose first, then run:

```sh
cd user_interface
npx playwright install chromium
STEGOLAB_BASE_URL=http://127.0.0.1:8081 STEGOLAB_RESTART_BACKEND=1 npm run test:browser
```

This browser check changes and resets settings on the local test installation
and restarts its backend. Do not run it against settings you need to preserve.
The browser test does not read `infrastructure/.env`, so set `STEGOLAB_BASE_URL`
explicitly as shown. Use your chosen port if it differs from 8081. Without this
variable, the test uses port 8080. Compose still reads the saved port for its
backend restart; no shell export of `STEGOLAB_PORT` is needed.

For native application development, start the backend in one terminal and the
frontend in another. The development server proxies API requests to port 8000.

```sh
uv run uvicorn backend_service.application:create_application --factory --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 15 --no-access-log
```

```sh
npm --prefix user_interface run dev -- --host 127.0.0.1
```

## Command line

The command line uses the same configuration schema and store as the API:

```sh
uv run stegolab --help
uv run stegolab inspect_configuration
uv run stegolab validate_configuration /path/to/configuration.json
uv run stegolab verify_protocol
uv run stegolab inspect_image /path/to/cover.jpg
uv run stegolab prepare_image /path/to/cover.jpg /path/to/new_prepared.png
```

A configuration document contains `schema_version: 1` and
`checkpoint_interval_seconds: 300`. The interval is a positive integer divisible
by 60. Validation reads the file without saving it. Native commands use `.runtime/`
unless `STEGOLAB_DATA_DIRECTORY` selects another location; this is separate from
the Docker volume. Experimental `train`, `evaluate`, `export_models`, and
`inspect_checkpoint` commands use the model workflow's explicit requests and
output directories. There are no `stegolab encode` or `decode` commands; each installed package
ships `runtime.py encode|decode`.

The protocol demonstration uses packaged public fixtures and accepts no secret
inputs. Image commands return safe metadata; preparation refuses existing output
files and preserves prepared integer pixels. These commands do not initialize
SQLite or change saved configuration. See the [protocol guide](message_protocol.md)
and [image guide](image_preparation.md) for supported inputs and exact rules.
Inside a running container, the public demonstration is also available through
`python -m backend_service.command_line verify_protocol`.

## Troubleshooting

| Problem | Check or action |
|---|---|
| Docker cannot connect | Start Docker Desktop, then retry `docker compose ... up`. |
| The selected port is busy | Choose a free port in `infrastructure/.env` and run the start command again. Leave other applications running. |
| Compose uses the wrong port | Run `unset STEGOLAB_PORT` to clear a shell override, then run the start command again. |
| The page shows disconnected | Check `docker compose -f infrastructure/compose.yaml ps` and the logs below. |
| Settings cannot save | Check disk space and volume permissions. The existing saved value is preserved on a failed write. |
| Startup reports invalid saved settings | Keep the volume for diagnosis; do not delete it or overwrite its database. Restore a known-good backup. |
| The image build fails | Confirm network access to image and package registries, then retry the locked build. |

```sh
docker compose -f infrastructure/compose.yaml logs --tail 100 backend_service user_interface
curl --fail http://127.0.0.1:8081/api/v1/health
```

Use your chosen port in the health URL; the shared default without an override
is 8080.

Base images are pinned to multi-platform digests in each Dockerfile. Updating a
digest or dependency lock requires running these checks and the restart test.
The CI workflow runs the same CPU checks and small synthetic model fixtures.
Long learning experiments stay outside routine CI.
