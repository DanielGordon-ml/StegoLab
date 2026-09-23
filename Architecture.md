# StegoLab architecture

StegoLab is a single-user workspace for training and using a text-in-image
steganography system. Its target flow is `(image, text, password) → PNG` and
`(PNG, password, compatible decoder) → exact text`. The decoder will not need
the original image. Training and deployment packages are separate concerns.

**Current state:** Sprint 1, Sprint 2, and the local dataset slice of Sprint 3 are implemented. The app can save
settings, package and recover authenticated messages through a test channel,
and prepare images without changing the prepared pixels during PNG storage.
Local UHD-IQA preparation adds grayscale support, source splits, exact duplicate
groups, immutable manifests, safe storage, and offline validation. It cannot yet
hide text in images. Models, remote dataset downloads, training workers, GPU
runtime, and cloud deployment remain unimplemented.

See the [system diagram in the README](README.md#system-architecture) and the
[interactive Archify map and delivery record](docs/architecture.md).
The sections below distinguish **implemented** components from the **planned**
first release and later research.

## 1. Evidence and decision order

This document reviews the code and all nine Markdown files in `plan/`.
Existing source and tests define current behavior. The approved backend,
frontend, infrastructure, and execution plans define the target. Sprint records
describe delivered scope; original briefs and the context document explain its
history. A plan entry is not evidence that a feature exists.

| Source | Role |
|---|---|
| [Backend plan](plan/backend.md) | Target services, protocol, data, models, training, exports, and release gates |
| [Frontend plan](plan/frontend.md) | Four-tab workflows, contracts, progress, privacy, and accessibility |
| [Infrastructure plan](plan/infrastructure.md) | Two-container target, private GPU host, persistence, recovery, and budget |
| [Execution order](plan/execution_order.md) | Dependencies, pilot allocations, and release sequence |
| [Sprint 1](plan/sprint_01.md), [Sprint 2](plan/sprint_02.md) | Implementation decisions and historical acceptance evidence |
| Local notes: `plan/sprints/sprint1.md`, `plan/sprints/sprint 2.md` | Earlier scope proposals; the completed sprint records take priority |
| Local note: `plan/CONTEXT.md` | Product intent and early ideas; later plans refine its promises |

The three local notes were reviewed in this workspace. They are untracked user
files and are not included in a fresh checkout; their relevant decisions and
later corrections are summarized here.

The context draft's character limits, continuous strength slider, TorchScript
exports, top-three-only checkpoint retention, fixed stop/cancel deadlines, and
SOTA claims are not the final contract. The later plans use UTF-8 **bytes**,
tested profiles, `torch.export`, latest recovery plus three best and pinned
checkpoints, safe completion boundaries, and measured quality gates. Random
binary payloads train the first model; a text corpus is optional test data.
Historical sprint counts and “no CI run” notes describe those implementation
sessions, not the current remote release state.

## 2. System boundaries and component responsibilities

The first release keeps two application containers. It does not require Redis,
Kubernetes, a distributed scheduler, or a multi-user account service.

| Boundary | Implemented today | Planned extension |
|---|---|---|
| Browser | React/TypeScript shell, Encode/Decode/Train/Config tabs, working Config, connection state | Image transfer, jobs, training charts, model selection, temporary text results |
| Frontend container | Nginx serves bundled Vite assets and proxies `/api/v1` on the same origin | Upload handling and long-lived event streaming within coordinated limits |
| Backend container | One Uvicorn/FastAPI process; CLI and reusable CPU services | Supervisor, scheduler, spawned GPU/download workers, model services |
| Shared contracts | Strict Pydantic records, dataset requests/manifests, exported schemas, browser validation | Artifact, model, checkpoint, and evaluation records |
| Persistent volume | SQLite configuration/retry/job metadata; structured run logs; prepared dataset revisions | Download cache, checkpoints, exports, artifacts, durable job events |
| Independent packages | None yet | Separate encoder and decoder exports with their own runtime wrappers |
| External sources | Read-only local image folders and UHD-IQA metadata | Browser uploads, Hugging Face, and controlled HTTPS downloads |

Current request flow is browser → frontend/proxy → API → state store.
Protocol, image, and local dataset services are reached through Python or the CLI;
there is no HTTP connection from the browser to those services yet.

### Repository map

| Path | Responsibility |
|---|---|
| [`user_interface/src/`](user_interface/src/) | Components, styles, transport adapters, runtime validators, and component tests |
| [`backend_service/application.py`](backend_service/application.py), [`routes.py`](backend_service/routes.py) | Startup checks, service wiring, thin API routes, and response schemas |
| [`backend_service/storage.py`](backend_service/storage.py) | Versioned SQLite state, transactions, retry results, and job snapshots |
| [`backend_service/message_protocol.py`](backend_service/message_protocol.py), `message_frame.py`, `message_correction.py`, `payload_capacity.py`, `payload_map.py` | Encryption/framing, correction, capacity accounting, and test payload maps |
| [`backend_service/image_preparation.py`](backend_service/image_preparation.py), `image_validation.py`, `image_color.py`, `image_png_stream.py`, `image_output.py` | Bounded input, content validation, color/orientation, integer PNG pixels, safe output |
| [`backend_service/command_line.py`](backend_service/command_line.py) | Small headless interface over shared services |
| [`schemas/`](schemas/), [`contracts/`](contracts/) | Structural source records and deterministic exported contracts |
| [`infrastructure/`](infrastructure/), [`backend_service/Dockerfile`](backend_service/Dockerfile), [`user_interface/Dockerfile`](user_interface/Dockerfile) | Compose runtime and separate locked CPU image builds |
| [`tests/backend/`](tests/backend/), [`user_interface/browser_tests/`](user_interface/browser_tests/), [`scripts/`](scripts/) | Service/contract/browser checks, independent fixtures, resource measurements |
| `datasets/`, `models/` | Prepared immutable dataset revisions; deployment models remain future work |
| [`plan/`](plan/), [`docs/`](docs/) | Product/delivery plans and user/developer guides |

## 3. Frontend, API, and contract flow

**Implemented:** React, TypeScript, Vite, and TanStack Query use small transport
adapters. Health and capabilities refresh every 30 seconds. Config accepts
positive whole minutes and persists seconds; its default is 300 seconds.
Save and Reset confirm success only after the backend response. The other three
tabs show unavailable states. Public capabilities report CPU, no models or
profiles, zero payload bytes, and encoding/decoding/training disabled.

Pydantic `StrictRecord` rejects unknown fields and unwanted type conversion.
`backend_service.export_contracts` exports OpenAPI plus entity JSON Schemas;
browser types and adapters interpret those schemas with `@cfworker/json-schema`.
The generated-client proof exceeded the small-file rule, so this documented
fallback is intentional. CI detects schema drift. Internal protocol/image
schemas are exported without creating HTTP routes.

| Implemented route under `/api/v1` | Behavior |
|---|---|
| `GET /health` | Readiness after startup and saved-state validation |
| `GET /capabilities`, `GET /models` | Actual available features and empty model list |
| `GET /configuration` | Read saved settings |
| `PUT /configuration`, `POST /configuration/reset` | Transactional settings changes with client request identifiers |
| `GET /jobs`, `GET /jobs/{job_identifier}` | Read stored metadata; these do not start work |

**Planned:** image registration and capacity routes; encoding, decoding,
training, evaluation, dataset, checkpoint, and export jobs; action requests;
registered artifact downloads; versioned configuration import/export; and one
ordered server-sent event (SSE) stream. The backend supplies job actions and
tested profile limits. A request identifier makes uncertain mutation retries
safe. Running jobs keep a frozen resolved configuration.

The browser will count UTF-8 bytes with `TextEncoder`, display the prepared
cover, and download the exact backend PNG without canvas redraw or recompression.
Decoded text will stay outside the shared query cache and browser storage.
Reconnection will deduplicate events, replay by cursor, or reload snapshots.
Polling every five seconds is a planned fallback only while disconnected with
active work. Keyboard access, labels, visible focus, and text beside status
colors remain required; automated checks alone are not a full accessibility audit.

## 4. Message path and model boundary

### Implemented protocol path

1. Validate trusted dimensions/profile context, password, and exact UTF-8 bytes.
2. Frame the message length, message bytes, and zero padding; encrypt the body.
3. Protect the full header and ciphertext with Reed–Solomon blocks; interleave.
4. Expand bits most-significant first into a repeated one-channel payload map.
5. In the controlled test channel, recover bits from hard values or finite logits.
6. Undo interleaving, correct blocks, authenticate, validate length/padding/UTF-8,
   and only then return the original text.

| Protocol rule | Version 1 behavior |
|---|---|
| Text/passwords | Strict UTF-8, no trimming/normalization; empty text allowed; passwords contain 1–1,024 bytes |
| Key derivation | Argon2id; 32-byte key, 64 MiB memory, operation limit 3; incoming data cannot change costs |
| Encryption | XChaCha20-Poly1305; independent fresh 16-byte salt and 24-byte nonce |
| Header/context | `SGLB`, one-byte version, salt, nonce; authenticated domain/header, length-prefixed trusted compatibility identity, 32-bit big-endian dimensions |
| Capacity/frame | `C = min(1024, floor(width × height / 1024))`; `B = ceil((C + 63) / 223)`; frame is `223 × B` bytes including the tag |
| Correction | 223 data + 32 parity bytes; GF(256), polynomial `0x11d`, generator 2, first root 0; byte-column interleaving |
| Payload map | `[1, height, width]`, row order, repeated protected bits; float64 sums of float32 logits, positive means 1 and ties mean 0 |
| Validity | Authentication decides validity; up to 16 damaged byte symbols per correction block can be corrected |

`test_only_v1` proves this layout, not a usable model capacity. At 1024 × 1024,
1,024 user bytes become a 1,115-byte frame and 10,200 protected bits repeated
across 1,048,576 map positions. Net user capacity, protected bits, and one raw
map bit per pixel are separate quantities. Beyond the correction guarantee,
only exact authenticated recovery or safe failure is acceptable.

The [protocol guide](docs/message_protocol.md) freezes the complete byte layout.
Four [independently generated public examples](docs/protocol_fixture_provenance.md)
check compatibility. Production calls have no deterministic-randomness switch.

### Planned image-to-message integration

The encoder receives prepared RGB plus the payload map and returns RGB.
The application clamps and rounds to eight-bit pixels, preserves oriented alpha,
saves PNG, then reopens it and runs the matching decoder before enabling download.
The decoder returns logits; the existing protocol reverses correction and
encryption. The decoder requires a compatible model/profile and password, not
the original cover or encoder weights. Crypto, file handling, and the download
verification step remain outside tensor-only model graphs.

## 5. Image path and pixel guarantees

Implemented services accept single-frame, eight-bit RGB JPEG and RGB/RGBA PNG.
They detect content rather than trust the filename. Limits are 50 MiB per input,
512–4096 pixels per side, and 8,850,000 pixels total. Both 3840 × 2160 and
4096 × 2160 fit; 4096 × 4096 does not. Header, byte, and compressed-stream checks
precede full pixel allocation. Unknown-length streams are bounded too.

Preparation applies EXIF orientation once, converts supported RGB ICC profiles
to sRGB, and preserves alpha after orientation. Explicit sRGB is accepted;
untagged RGB records an sRGB assumption. Unsupported modes, bit depths, animation,
color-key transparency, or color declarations fail with conversion guidance.
There is no silent crop or resize.

RGB remains RGB and RGBA remains RGBA. Source metadata is stripped. Output goes
to a same-directory temporary file, is flushed, reopened, and compared channel
by channel, then published atomically with a no-overwrite hard link. Existing
files survive failure; the destination filesystem must support hard links.
`read_prepared_png` validates PNG but performs no orientation or color conversion:
it preserves stored integer pixels. This proves PNG storage, not neural hiding.
See [image preparation](docs/image_preparation.md) for exact rules and commands.

## 6. State, jobs, and file ownership

**Implemented:** SQLite schema version 1 uses WAL and short transactions.
Its tables are `configuration`, `mutation_results`, and `jobs`. Configuration
and its retry result commit together; conflicting reuse of a request identifier
fails. Saved state is validated on read. Corrupt/unsupported state fails safely
without silently replacing it with defaults. `JobSnapshot` and `JobEvent`
contracts exist, but no worker, scheduler, event table, or live event route does.

**Planned:** the backend supervisor owns spawned GPU and download processes.
The API/scheduler alone changes durable job state; workers send progress,
heartbeats, and results over bounded in-memory channels. Heavy work stays out
of the HTTP event loop. One GPU operation runs at a time; inference queues
behind training/evaluation. A lost heartbeat does not free that slot until the
supervisor confirms the old process has stopped.

Job states are `queued`, `running`, `paused`, `stopped`, `completed`, `cancelled`,
`failed`, `interrupted`, and `needs_input`. Preparation, validation, saving, and
cleanup are phases. Browser disconnects do not stop jobs. Restart reconciliation
requires explicit training resume and fresh inputs for secret-bearing inference;
paid training must never restart automatically.

| Record/storage class | Ownership and lifetime |
|---|---|
| Configuration, job snapshots, request retry results | SQLite now; future durable events remain metadata only |
| `ProtocolContext`, `PayloadCapacity`, `ProtocolVerification`, `ImageSummary` | Implemented strict service records; no message/password fields |
| Dataset manifests | Implemented strict records, software/source provenance, checksums, fixed splits, and coverage |
| Model manifests, checkpoint summaries, evaluation reports, artifacts | Planned strict records with compatibility, provenance, and checksums |
| Prepared datasets/cache | Implemented `datasets/<name>/<revision>/`; download cache remains planned |
| Models/checkpoints | Planned separate deployment and training directories; checkpoint pruning never deletes deployment packages |
| Uploads/previews/PNG results | Planned registered files, expiring after 24 hours or explicit deletion; never arbitrary path downloads |
| Plaintext/passwords/keys | Temporary memory only; decoded text planned for five minutes or explicit Clear, with `Cache-Control: no-store` |
| Logs | Current bounded `events.jsonl` files in unique date-labelled run directories; future run metrics add approved metadata only |

## 7. Local datasets and planned training, evaluation, and exports

The implemented local pipeline scans bounded sources, reads UHD-IQA metadata,
preserves official splits, prepares grayscale/RGB/RGBA images, groups duplicate
RGB pixels and shared upstream identities, and publishes verified revisions.
Dataset limits are separate from production: 1–8192 per side, 32 million pixels,
50 MiB source files, and 128 MiB prepared PNGs. A destination-root lock protects
the disk budget and owned staging; completed revisions work without sources.
The CLI exposes `prepare_dataset`, `inspect_dataset`, and `validate_dataset`.
The offline reader supplies eligible unique examples from one requested split.
See [the dataset guide](docs/datasets.md) for contracts and failure behavior.

Future adapters cover browser uploads, Hugging Face sources, and supported HTTPS
files/archives. Imports must not execute remote scripts. Connection-time and redirect
checks block private/link-local/metadata destinations; extraction limits block
traversal, escaping links, and decompression bombs. Pause depends on source
support; cancel removes only that job's partial assets.

Train/tuning/test source identities are frozen before crops. Exact duplicate
checks are implemented; a near-duplicate audit is required before the pilot.
Cover augmentations happen before embedding. Post-embedding channel
attacks are a separate later profile. COCO supplies the main planned benchmark;
native DIV2K/high-resolution tests are reported separately. Dataset mirrors,
splits, licenses, and revisions are not interchangeable.

The baseline is a modern PyTorch dense/residual encoder and decoder: four 3×3
convolution stages, 32 hidden channels, dense concatenation, hidden LeakyReLU and
BatchNorm, three-channel encoder residual, and one decoder-logit channel.
BatchNorm freezes for inference. The optional critic and channel simulator are
replaceable training components, absent from independent deployment packages.
The first experiment uses random bits, 256-pixel crops, FP32, Adam at `1e-4`,
effective batch 16, recorded seeds, bit loss plus scheduled image loss, and a
documented gradient approximation through eight-bit rounding.

Evaluation uses saved/reopened integer PNGs. It records exact-message recovery,
raw bit error rate, PSNR, SSIM, LPIPS, clipping, latency, memory, and failures.
Full-frame and halo-tiled inference must agree at borders and odd dimensions.
The release target is at least 99.9% observed exact recovery across 10,000 frozen
1024 × 1024 trials, each with 1,024 user bytes, median PSNR ≥40 dB, and median
SSIM ≥0.98. These are unproven targets. Results before the download verification
filter, retries/refusals, and uncertainty must be reported.

Full checkpoints include networks, optimizer/scheduler/scaler, random and sampler
state, consumed steps, manifests, frozen configuration, and environment/source
identifiers. Save at safe step boundaries, verify before publication, then prune;
keep latest recovery, three best, and pinned checkpoints. Exact continuation is
limited to the same supported deterministic environment.

Separate `torch.export` encoder and decoder packages will include small wrappers,
manifests, checksums, shape/capacity rules, protocol/preprocessing versions,
dependencies, and known-answer vectors. Each must load independently in clean
CPU and GPU processes. Detection resistance, robust channels, larger capacities,
and SOTA comparisons require later independent research and compute budgets.

## 8. Deployment, privacy, and failure boundaries

**Current CPU deployment:** [`infrastructure/compose.yaml`](infrastructure/compose.yaml)
publishes only `127.0.0.1:${STEGOLAB_PORT:-8080}` from the frontend container.
The backend listens at internal port 8000; SQLite is not a network service.
The known local installation uses port 8081. The named `application_data` volume
mounts at `/data`; the backend uses `/data/state` and `/data/logs`. Native commands
default to `.runtime` state and `logs` unless environment settings override them.

Both images run as non-root with read-only root filesystems, dropped capabilities,
no privilege escalation, health checks, and bounded Docker logs. Backend/frontend
host-memory limits are 10 GiB/256 MiB and `/tmp` limits are 64/32 MiB. Nginx
currently permits 1 MiB request bodies. These foundation settings do not yet
support the planned 50 MiB browser upload workflow. Dependencies and base-image
digests are locked; application version remains `0.1.0`.

**Planned GPU deployment:** a separate Linux amd64 CUDA target runs on one private
EC2 NVIDIA instance; the plan selects g6.2xlarge and an image resolved/pinned for
the chosen region. A 60 GiB encrypted root volume and 200 GiB encrypted gp3 data
volume support the pilot. The browser connects through SSH, with no public app
port. The app receives no AWS credentials/role; optional Hugging Face read tokens
are protected files. GPU availability, exports, restores, and the stop guard must
be checked before paid training.

The 24 allocated GPU-instance hours include startup, idle time, training,
evaluation, saves, and shutdown across all sessions. A persistent ledger and
independent host deadline guard prevent resets from renewing the budget; an
external scheduled stop is the planned unattended backstop. Keep at least
10 GiB or 10% disk space free, whichever is larger. Consistent SQLite backups and
paused file writes precede snapshots. Rollback selects a known-good app/model pair.

Secrets stay out of logs, database records, URLs, command arguments, environment
variables, filenames, image metadata, browser storage, and config exports.
The current CLI verification uses public bundled fixtures and accepts no secrets.
Protocol services perform no persistence. Python reference release is not a
guarantee of memory erasure. Future worker process separation improves recovery
and responsiveness; it is not a hostile-code sandbox.

Safe error records carry a code, plain explanation, and opaque diagnostic
reference. API errors and logs exclude submitted values and raw tracebacks;
image services suppress source-bearing decoder diagnostics. Wrong-password,
missing, and damaged messages share: “No valid hidden message could be recovered.
Check the password, model, and image.” Unsupported input, overflow, storage, and
resource failures remain separate. No guessed or unauthenticated partial text
may be returned.

## 9. Verification, delivery order, and remaining design work

The [CI workflow](.github/workflows/continuous_integration.yaml) runs locked
installs, formatting/lint/types, backend/frontend tests, public protocol vectors,
schema drift and file-length checks, CPU image builds, and a browser settings
save → backend restart → reload → reset flow on disposable data. Sprint 3 records
291 backend tests, 10 frontend tests, and the browser restart flow passing.
Dataset tests cover fixed splits, grayscale, duplicates, offline integrity, disk
limits, unsafe paths, interruption, and reuse. Image/protocol tests cover byte boundaries,
wrong context/password, correctable damage, malformed input, exact pixels, limits,
failed writes, and privacy. Expensive GPU benchmarks belong to promotion gates.

| Delivery stage | Dependency and remaining result |
|---|---|
| 1–2: foundation and protocol | Implemented; service/test foundations are available |
| 3: dataset adapters | Local preparation/manifests implemented; remote adapters/cache and complete pilot data remain open |
| 4: model/training engine | Depends on protocol and validated data; tiny-set learning, checkpoint resume, saved-PNG evaluation, CPU exports |
| 5: complete four-tab frontend | Build against contracts; final acceptance needs real models and artifacts |
| 6: EC2 readiness | Depends on CPU/export proof; CUDA, private access, recovery, ledger and shutdown drill |
| 7: capped pilot | 2 hours readiness, 14 baseline, 4 controlled critic ablation, 4 evaluation/integration; hard total 24 |
| 8: release integration | Promote only passing profiles; real encode/download/decode, datasets, restart, 4K and export acceptance |
| 9: SOTA research | Separate budget, independent detectors, fair classical/neural comparisons, robust profiles |

The target product choices are recorded, but implementation contracts still need
work: multipart transport and temporary-storage limits; registered artifact
ownership/expiry; worker channels, queue limits and crash reconciliation; durable
event replay/retention; remote dataset adapter and model/checkpoint compatibility;
temporary decoded-result handling; and CUDA/tiling resource policies. Neither
these services nor their future safety guarantees should be inferred from empty
directories, reserved CLI names, or existing metadata schemas.

Actual hiding, exact recovery from model-produced images, visual quality,
statistical detectability, and resistance to changed images are separate gates.
The current system establishes only the foundation, message protocol, and
prepared-PNG pixel preservation.
