# Sprint 2 — Message and Image Protocol

Status: implemented and locally verified. Core scope confirmed by owner.
Planned duration: ten working days. Team: project owner + Codex.
Date: 2026-09-23.

## Goal and scope

- Complete Step 2 of the [execution order](execution_order.md), building on
  [Sprint 1](sprint_01.md) and the decisions in [backend.md](backend.md).
- Convert text into a versioned, encrypted, error-protected payload and recover
  the exact original bytes through a controlled test channel.
- Prepare supported cover images and prove that saving and reopening their PNG
  representation preserves the prepared pixels and dimensions.
- These are separate proofs. Recovering bytes from a test payload does not prove
  that a model can hide or recover them from an image.
- Use CPU only. Dataset adapters, neural models, training, cloud deployment,
  production encoding/decoding, and image-quality benchmarks remain later work.
- Confirmed scope: backend services, shared contracts, a small inspection CLI,
  protocol fixtures, and documentation. Image-upload previews remain later work.

## Sprint backlog

| Days | Work | Acceptance |
|---|---|---|
| 1–2 | Freeze protocol version 1, image rules, capacity accounting, and test fixtures. Pin required dependencies. | Reviewed byte-layout specification and example vectors; both directions use the same rules. |
| 3–4 | Implement UTF-8 framing and password-based authenticated encryption. | Exact recovery for multilingual and boundary inputs; wrong password or changed context reveals no text. |
| 5–6 | Add error correction, interleaving, payload-map placement, and capacity checks. | Correctable damage recovers exact bytes; overflow and invalid layouts fail clearly. |
| 3–6, parallel | Implement image inspection, cover preparation, and the unchanged-PNG reader. | Supported images retain prepared dimensions, RGB values, and supported alpha; bad inputs fail safely. |
| 7–8 | Integrate the services and CLI; review failures, privacy, compatibility, and documentation. | A repeatable local demonstration works; no secret values reach logs or saved state. |
| 9–10 | Run acceptance checks, fix defects, and record evidence. | Focused CI and the existing browser workflow pass; every acceptance item has a recorded result. |

- Codex owns integration and the protocol. An agent handles image preparation;
  a separate reviewer checks compatibility, fixtures, and failure handling.
- The owner reviews the protocol contract on Day 2 and the demonstration on
  Day 10. Changes after the contract review update the version and fixtures.
- Use separate branches such as `codex/sprint-02-message-protocol`,
  `codex/sprint-02-image-preparation`, and `codex/sprint-02-protocol-integration`.
- Keep the two-container foundation, naming rules, strict schemas, and source
  files below 300 lines. Do not repeat the completed client-generation proof.

## Protocol decisions

- Preserve UTF-8 bytes exactly, including Hebrew, emoji, combining characters,
  empty messages, spaces, and line breaks. Do not normalize or truncate text.
- Keep protocol versioning separate from configuration and application versions.
- Derive a 32-byte key with Argon2id, an independently random salt, 64 MiB memory,
  and operation limit 3. Use XChaCha20-Poly1305 through PyNaCl `Aead`, with a fresh
  24-byte nonce. Specify these parameters rather than taking library defaults.
  See [PyNaCl key derivation](https://pynacl.readthedocs.io/en/latest/password_hashing/)
  and [authenticated encryption](https://pynacl.readthedocs.io/en/latest/secret/).
- Authenticate the protocol version and compatible model/profile context.
  Reject unsupported profiles and unreasonable lengths before expensive work;
  incoming bytes must not select arbitrary key-derivation resource limits.
- Put the actual message length inside the authenticated content. Pad to the
  frame size selected by image dimensions and protocol profile, so a future
  decoder can locate the frame without already knowing the secret length.
- Use Reed-Solomon blocks with 223 data bytes and 32 parity bytes. The Day 2
  specification must freeze field parameters, block padding, byte/bit order,
  interleaving, repeated-bit positions, unused positions, and recovery rules.
  Protect the complete frame, including the header needed for recovery.
- Define payload-map dimensions and reconstruction from decoded bits or logits.
  Reconstruction must use only the image dimensions and selected profile, with
  no original cover, encoder weights, or per-message side file.
- Correction happens before authentication. Authentication must succeed before
  any text is returned. Error correction alone cannot establish message validity;
  see the [codec's correction limits](https://github.com/tomerfiliba-org/reedsolomon).
- For absent, damaged, or unauthenticated payloads, use the same recovery error:
  "No valid hidden message could be recovered. Check the password, model, and
  image." Keep format/resource errors separate; never include submitted values
  or raw tracebacks.
- Production randomness has no deterministic mode. Fixed salt/nonce values are
  allowed only in clearly labelled, public test fixtures.
- Keep messages, passwords, and derived keys in short-lived memory. Do not send
  them to logs, SQLite, filenames, command arguments, environment variables,
  image metadata, or diagnostic output. Release references after use; do not
  claim guaranteed memory erasure in Python.

## Capacity decisions

- Compute an internal candidate budget of
  `min(1024, floor(width * height / 1024))` net message bytes, within image limits.
- Account separately for user bytes, framing, salt/nonce/authentication overhead,
  padding, correction bytes, and repeated bits. Reduce or reject a candidate
  layout if its complete payload cannot fit. Reject over-capacity input early.
- Test representative 512 × 512, 1024 × 1024, odd-sized, and 4K layouts, including
  exact maximum and one-byte-overflow cases. Record raw and net capacity apart.
  Include 3840 × 2160 and 4096 × 2160; reject 4096 × 4096 on the area limit.
- This calculator proves layout fit only. Public capabilities keep zero usable
  payload capacity, empty model/profile lists, and encoding/decoding/training
  unavailable until a real model passes its release tests.
- Use a clearly named test-only profile for fixtures; it is never registered
  as an installed model or advertised as a usable encoding profile.
- Defer the public capacity endpoint and model selection. Never present the
  internal 1,024-byte target as an available encoding feature.

## Image decisions

- Accept single-frame, 8-bit RGB JPEG and RGB/RGBA PNG covers. Check actual file
  content and fully decode it; filename extensions are insufficient.
- For this version, reject animated images, palette/grayscale/CMYK modes,
  unsupported bit depths, and damaged files with clear conversion guidance.
- Apply EXIF orientation once and prepare supported color profiles into sRGB.
  Reject malformed or unsupported color profiles; treat untagged RGB as sRGB
  with a recorded assumption. Strip source metadata from the prepared output.
- Preserve dimensions after orientation, without resizing or cropping. Each
  side must be 512–4096 pixels and total area at most 8,850,000 pixels.
- Preserve supported alpha separately and unchanged after orientation. Models
  will operate on RGB only. Document RGB/RGBA output handling explicitly.
- The decode-side PNG reader must preserve integer pixels: no orientation,
  resize, or color conversion. It returns validated pixels without claiming to
  find a message.
- Cap input files at 50 MiB, check dimensions before full allocation, and handle
  decompression-limit failures as safe errors. Enforce limits during reading,
  including inputs without a trusted size. Keep Pillow's protections enabled;
  see [Pillow image loading](https://pillow.readthedocs.io/en/stable/reference/Image.html).
- Write CLI outputs atomically; do not overwrite an existing file by default.
  Failed preparation must leave the source and previous output unchanged.

## Interfaces and demonstration

- Put protocol/image services in `backend_service/` and strict structural
  records in `schemas/`. Keep cryptography and file handling outside future
  tensor-only model graphs. No new worker scheduler or event stream is needed.
- Add CLI image inspection/preparation and a verification command using bundled
  public protocol fixtures. Reuse service code; report pass/fail and safe metadata.
  The verification command does not accept real passwords or secret messages.
- Keep `encode`, `decode`, and `train` unavailable. The four tabs and working
  Config screen keep their current behavior.
- Export updated entity schemas/OpenAPI where applicable and keep browser
  validators aligned. Internal services do not require public HTTP endpoints.
- Demo: inspect a supported image → prepare PNG → reopen and verify pixels;
  run public multilingual fixtures → recover exact bytes → demonstrate wrong
  password, corrected damage, rejected invalid data, and capacity overflow.
- Write `docs/message_protocol.md` with the byte layout and fixture provenance,
  plus image rules and repeatable commands in `docs/image_preparation.md`.
- Update the existing Archify map only where implemented components change;
  keep models/workers marked planned and repeat Archify validation and visual
  checks for any changed artifact.

## Deferred GUI work

- Image upload and server previews need registered artifacts, retry handling,
  expiry/deletion, storage limits, and restart cleanup. Deliver that complete
  slice in a later sprint; no upload/artifact routes are added here.
- Carry forward the integration findings: the browser adapter handles JSON,
  the proxy currently limits requests to 1 MiB, and frontend/backend temporary
  storage is 32/64 MiB. Multipart uploads will need coordinated limits, bounded
  processing, and their own acceptance tests.

## Definition of done and acceptance record

The implementation and evidence below use CPU only. All acceptance checks passed
locally; protocol fixtures were generated and reviewed independently.

| Gate | Required result | Status |
|---|---|---|
| Protocol specification | Versioned layout, parameters, context, padding, placement, and fixture provenance are documented and reviewed. | Passed; exact wire rules in the protocol guide |
| Known-answer checks | Committed expected bytes match reviewed vectors; tests do not regenerate their expectations through the implementation under test. | Passed; four independently generated vectors |
| Exact text | Empty, multilingual, whitespace, and maximum-size messages recover byte-for-byte; overflow and invalid UTF-8 fail clearly. | Passed |
| Authentication | Absent/damaged/unauthenticated payloads use the same safe recovery error and return no partial text; unsupported versions fail without unsafe resource use. | Passed |
| Error correction | Up to 16 corrupted byte symbols per 255-byte block recover; beyond the guarantee, accept only exact authenticated recovery or a safe failure. | Passed; includes header, data, and parity across five blocks |
| Capacity | Every tested layout accounts for all overhead; invalid dimensions and one-byte overflow are rejected. Public capacity remains zero. | Passed; includes every block-count transition and both 4K sizes |
| Images | Orientation, color, alpha, limits, malformed files, odd dimensions, and prepared-PNG pixel equality pass. | Passed; 89 image-service tests plus CLI coverage |
| Privacy and writes | Secret-like test values are absent from captured logs, state, and output metadata; failed writes preserve previous files. | Passed; includes dependency DEBUG logs and allocation failures |
| Compatibility | Existing settings/job metadata survive; strict contracts and the Sprint 1 browser restart workflow still pass. | Passed; actual backend restart on disposable data |
| Delivery | Locked CPU builds, formatting/types, focused tests, schema checks, and code-length checks pass; CLI demo and docs work from a fresh checkout. | Passed; local results below |

- Tests inject damage into byte buffers or payload maps. Do not describe this as
  JPEG, resize, or image-transmission robustness.
- Measure CPU time and peak memory for representative cases; record results
  without promising universal latency or memory safety.
- Use synthetic/public fixtures only. Keep large generated images and runtime
  output outside Git. No GPU hours or SOTA claims are part of this sprint.
- After this gate, Sprint 3 can deliver local dataset preparation and manifests;
  the model/training step depends on both the protocol and validated data.

## Implementation and acceptance evidence

- Source verified at `370cbc9` on `codex/sprint-02-protocol-integration`.
  Later edits only update documentation and this acceptance record.
- Feature work used separate protocol, image, fixture, and architecture branches.
  Independent review checked frame bytes, correction boundaries, image validation,
  log privacy, and failure cleanup. All reported defects were fixed and rechecked.
- Added `inspect_image`, `prepare_image`, and `verify_protocol`. Verification
  accepts no secret arguments and uses only packaged public examples.
- Added strict `ProtocolContext`, `PayloadCapacity`, `ProtocolVerification`, and
  `ImageSummary` entity schemas. No HTTP route, database migration, model
  registration, or frontend behavior changed.
- Protocol, image preparation, and failure rules are documented in
  [message protocol](../docs/message_protocol.md),
  [image preparation](../docs/image_preparation.md), and
  [fixture provenance](../docs/protocol_fixture_provenance.md).
- Locked new runtime dependencies: PyNaCl 1.6.2, Pillow 12.3.0, reedsolo 1.7.0,
  and NumPy 2.5.3. NumPy supplies CPU arrays and stable float64 logit accumulation;
  this sprint adds no PyTorch dependency.

| Command or check | Local result |
|---|---|
| `uv run python -m pytest tests/backend` | 192 passed |
| `npm --prefix user_interface test` | 10 passed |
| Ruff lint and formatting; mypy including backend tests | Passed; 50 files type-checked |
| Frontend TypeScript, ESLint, Prettier, and Vite production build | Passed |
| `python -m backend_service.export_contracts --check` | Passed; existing OpenAPI routes unchanged |
| `python scripts/check_file_lengths.py` | 81 source files, all below 300 lines |
| `uv build --wheel` | Passed; fixtures and new modules included |
| Isolated wheel import and `verify_protocol` | Passed; no reliance on checkout fixture paths |
| Clean-checkout CPU Compose build and startup | Passed on Linux arm64, both services healthy |
| Container protocol verification and image inspect → prepare → reopen | Passed |
| Playwright save → restart → reload → reset | 1 passed; no page errors or automated accessibility violations |
| Archify artifact, browser, and visual review | Passed; see [delivery record](../docs/architecture.md) |

- CPU container acceptance used a clean detached worktree at `370cbc9`, project
  `stegolab_sprint02_acceptance`, port 8082, and a new disposable named volume.
  Its containers, volume, network, and checkout were removed after verification.
- The browser command used `COMPOSE_PROJECT_NAME=stegolab_sprint02_acceptance`,
  `STEGOLAB_PORT=8082`, `STEGOLAB_BASE_URL=http://127.0.0.1:8082`, and
  `STEGOLAB_RESTART_BACKEND=1`. The original app on port 8081 stayed available
  with its saved 300-second checkpoint interval.
- The updated CI workflow also runs the public protocol demonstration. These
  checks ran locally; no GitHub Actions run or cloud deployment was triggered.
- The existing Starlette/httpx test-client deprecation warning remains
  non-blocking. No GPU hours were used.

## CPU time and memory observations

Run `uv run python scripts/measure_protocol_resources.py` from the repository root.
Each case uses a separate process, a full-capacity public message, and a synthetic
patterned RGBA PNG. It verifies protocol recovery and exact prepared pixels.
Image timings include preparation, writing, reopening, and pixel comparison;
source-fixture generation is outside that timed interval.

Measured on 2026-09-23: macOS 26.6.2 arm64, Python 3.12.13, locked dependencies.
Peak resident memory includes imports, fixture buffers, and both workloads.

| Dimensions | User bytes | Protocol wall / CPU seconds | Image wall / CPU seconds | Peak process MiB |
|---|---:|---:|---:|---:|
| 512 × 512 | 256 | 0.9610 / 0.9542 | 0.0151 / 0.0132 | 107.77 |
| 1024 × 1024 | 1,024 | 0.9668 / 0.9616 | 0.0384 / 0.0383 | 109.25 |
| 4096 × 2160 | 1,024 | 0.9540 / 0.9513 | 0.3125 / 0.3113 | 334.94 |

These measurements are observations on public synthetic inputs, not universal
latency or memory guarantees. Actual image-message recovery, visual quality,
detection resistance, and SOTA evaluation remain future model work.
