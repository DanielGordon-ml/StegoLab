# StegoLab

A local workspace for building a text-in-image steganography system.

The application provides four accessible tabs, a working Config screen, strict
backend contracts, persistent settings, and CPU containers. Sprint 2 adds
encrypted message framing, error correction, test payload maps, and pixel-safe
image preparation through reusable services and command-line tools.
Image-based encoding, decoding, training, and dataset downloads remain unavailable.
No trained model or measured image-message recovery quality is included.

## System architecture

The map covers the full planned system. **Solid lines** show implemented paths;
**dashed lines** show planned work. The target keeps two containers: frontend
and backend. Future model and data workers run as processes inside the backend.

[![StegoLab architecture: browser and CLI, frontend and backend, message and image services, planned dataset and model workers, persistent state, checkpoints, and independent model packages](docs/architecture.svg)](Architecture.md)

Read [Architecture.md](Architecture.md) for component responsibilities, message
and image flows, contracts, storage, security, training, deployment, and roadmap.
The [Archify guide](docs/architecture.md) links the interactive local map and
explains how to reproduce both diagrams. Only Config and the local protocol/image
tools work today; model quality and real image-message recovery remain unmeasured.

## Start locally

Install Docker with Compose, then run from the repository root:

```sh
docker compose -f infrastructure/compose.yaml up --build --wait
```

Open [StegoLab](http://127.0.0.1:8080). In Config, change the checkpoint frequency,
save, restart the backend, and reload the page. The saved value remains available.
Reset restores five minutes.

If port 8080 is already in use, start with
`STEGOLAB_PORT=8081 docker compose -f infrastructure/compose.yaml up --build --wait`
and open [StegoLab on port 8081](http://127.0.0.1:8081). Use the same environment
variable for later Compose commands.

```sh
docker compose -f infrastructure/compose.yaml restart backend_service
docker compose -f infrastructure/compose.yaml down
```

Stopping containers preserves settings in the named volume. Do not add `--volumes`
unless you intend to delete that saved state.

## Project guides

- [Full system architecture: implemented and planned](Architecture.md)
- [Local setup, checks, and troubleshooting](docs/local_setup.md)
- [Sprint 1 backlog and acceptance results](plan/sprint_01.md)
- [Sprint 2 backlog and acceptance results](plan/sprint_02.md)
- [Message protocol and public demonstration](docs/message_protocol.md)
- [Image inspection and PNG preparation](docs/image_preparation.md)
- [Interactive architecture map and reproduction steps](docs/architecture.md)
- [Frontend binding proof](docs/client_binding_proof.md)
- [Backend plan](plan/backend.md), [frontend plan](plan/frontend.md),
  [infrastructure plan](plan/infrastructure.md), and
  [execution order](plan/execution_order.md)

The frontend is the only published service, bound to localhost. The backend stores
configuration and job metadata in SQLite. No cloud account or GPU is needed.
