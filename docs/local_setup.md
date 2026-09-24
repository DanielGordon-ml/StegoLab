# Run StegoLab locally

Sprint 1 provides four tabs and saved settings. Sprint 2 adds message-protocol
checks and image preparation through the command line. Sprint 3 adds local
dataset preparation and offline integrity checks; see [datasets](datasets.md).
Sprint 4 adds experimental CPU training, checkpoint inspection, evaluation, and
independent model exports through the [command-line workflow](model_training.md).
Browser training, encoding/decoding, dataset downloads, and model installation
remain unavailable. Public model capacity is still zero.

## Start with Docker

- Install and start Docker Desktop with Compose. Allocate enough memory for your
  other work; the backend has a 10 GiB upper limit and the frontend has 256 MiB.
- Run these commands from the repository root:

```sh
docker compose -f infrastructure/compose.yaml up --build --detach --wait
docker compose -f infrastructure/compose.yaml ps
```

Open [StegoLab](http://127.0.0.1:8080). Both containers must report healthy.
The build uses native CPU images on Apple Silicon and Linux. No GPU or cloud
account is needed. Only the frontend publishes a port, bound to your computer.

If another application uses port 8080, keep it running and choose another port:

```sh
export STEGOLAB_PORT=8081
docker compose -f infrastructure/compose.yaml up --build --detach --wait
```

Then open [StegoLab on port 8081](http://127.0.0.1:8081). Keep this environment
variable set for later Compose commands. The port always binds to localhost.

The Config tab starts with a five-minute checkpoint frequency. Save another
positive whole-minute value, restart the backend, then reload the page:

```sh
docker compose -f infrastructure/compose.yaml restart backend_service
```

The saved value must remain. Reset restores and saves five minutes. CPU and
model availability are read-only in this sprint.

## Storage and stopping

- The `stegolab_application_data` named volume stores settings and job metadata
  under `/data/state`, plus run logs under `/data/logs`.
- A new volume inherits the backend's non-root ownership during first startup.
  Do not replace it with a root-owned host directory.
- Docker output is limited to three 10 MiB files per service. Request access
  logs are disabled so request values do not enter proxy logs.
- Stop containers while keeping saved data:

```sh
docker compose -f infrastructure/compose.yaml down
```

Avoid `down --volumes` for your working installation: it deletes saved settings
and all other data in this volume. CI uses it only for disposable test data.

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
STEGOLAB_RESTART_BACKEND=1 npm run test:browser
```

This browser check changes and resets settings on the local test installation
and restarts its backend. Do not run it against settings you need to preserve.
Use `STEGOLAB_BASE_URL` to select another local test address.
For port 8081, run with `STEGOLAB_BASE_URL=http://127.0.0.1:8081` while keeping
`STEGOLAB_PORT=8081` set.

For native application development, start the backend in one terminal and the
frontend in another. The development server proxies API requests to port 8000.

```sh
uv run uvicorn backend_service.application:create_application --factory --host 127.0.0.1 --port 8000 --no-access-log
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
output directories. Public `encode` and `decode` commands remain unavailable.

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
| Port 8080 is busy | Set `STEGOLAB_PORT=8081` as described above. Leave other applications running. |
| The page shows disconnected | Check `docker compose -f infrastructure/compose.yaml ps` and the logs below. |
| Settings cannot save | Check disk space and volume permissions. The existing saved value is preserved on a failed write. |
| Startup reports invalid saved settings | Keep the volume for diagnosis; do not delete it or overwrite its database. Restore a known-good backup. |
| The image build fails | Confirm network access to image and package registries, then retry the locked build. |

```sh
docker compose -f infrastructure/compose.yaml logs --tail 100 backend_service user_interface
curl --fail http://127.0.0.1:8080/api/v1/health
```

Base images are pinned to multi-platform digests in each Dockerfile. Updating a
digest or dependency lock requires running these checks and the restart test.
The CI workflow runs the same CPU checks and small synthetic model fixtures.
Long learning experiments stay outside routine CI.
