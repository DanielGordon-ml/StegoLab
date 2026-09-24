# Sprint 4 — First CPU Model Proof

Status: implemented; all Sprint 4 acceptance gates passed; release qualification remains open.
Date: 2026-09-24. Planned duration: ten working days. Team: owner + Codex.
Branch: `codex/sprint-04-cpu-model-proof`. Source baseline: `5d83bd6`.

## Starting point and agreed outcome

- Sprints 1–3 are merged. Initial checks: 291 backend tests, 10 frontend tests,
  matching contracts, and 111 code files below 300 lines.
- The full local corpus has 6,073 prepared images, but this proof uses the frozen
  30-image sample revision
  `c3b9eb1bea77b4172a620814f047e5214d89b24ef53639582980546cfb20d0d2`.
- Goal: implement CPU learning, authenticated saved-PNG recovery, complete
  checkpoint resume, and independent encoder/decoder packages.
- Confirmed budget: four elapsed hours across at most two experiments; each
  experiment gets at most two hours including validation, saving, and exports.
  Reserve twenty minutes per experiment for saves and final checks. No GPU use.
- The scientific gate remains open if learning or a complete proof does not fit
  the budget. No weaker thresholds, hidden retries, or release/SOTA claims.

## Delivered implementation

- Python 3.12 with PyTorch 2.14.0 from an explicit CPU-only package index;
  native Mac arm64 and Linux CPU dependency locks. No torchvision.
- Dense encoder/decoder with four 3×3 convolution stages, 32 hidden channels,
  dense concatenation, LeakyReLU and BatchNorm, residual RGB output and one
  decoder logit channel. RGB uses float32 `[0,1]`.
- Fixed center 256×256 crops from the first four train/tuning representatives;
  no held-out learning or scoring. One complete dataset validation per
  invocation, deadline checks between files/crops, and a small-sample ceiling
  of 30 selected images and 512 MiB prepared bytes before pixel validation.
- Physical batch one, four accumulated microbatches, Adam `1e-4`, seed zero,
  random payload bits, and no data augmentation in this tiny-set proof.
- Mean bit BCE plus mean RGB MSE. Image weight is
  `100 * min(completed_optimizer_steps / 200, 1)` across 1,000 planned updates.
  The actual forward path clamps and rounds to 8-bit values; rounding uses
  a straight-through gradient in training.
- Validation every 100 steps, recovery saves every five minutes and at safe
  stop boundaries. Full network/BatchNorm/optimizer/random/sampler state,
  resolved settings, data/environment/code identities, atomic verification,
  and latest + three best + pinned retention.
- Persistent exclusive time ledger, conservative crash accounting, and safe
  retry after preflight failure without resetting spent time or slots.
- Separate `torch.export` packages with batch one, sides 512–1024, protocol/image
  wrappers, checksums, exact dependencies, and public known-answer vectors.
  Each package runs outside the repository with only its own network.

## Interfaces and boundaries

- Experimental CLI: `train <request.json>`, `evaluate <request.json>`,
  `export_models <request.json>`, and `inspect_checkpoint <directory>`.
- Strict shared records cover requests, configuration, runs, checkpoints,
  progress, evaluation, and exports. JSON requests can also arrive on stdin.
- Existing `test_only_v1` framing is unchanged; a deterministic pair checksum
  binds model compatibility. Real message tests remain at least 512 pixels.
- The decoder runtime needs no cover or encoder. Application verification of
  downloadable PNGs is outside the independent encoder package.
- HTTP routes and public capabilities remain unchanged: no installed model,
  zero usable capacity, and no GUI encoding, decoding, or training.
- Deferred: download adapters, workers/events, GUI integration, fine-tuning,
  critic/attack layers, tiling, 4K, GPU/cloud work, and release promotion.

## Acceptance gates

| Gate | Required evidence |
|---|---|
| Tiny model learning | Mean raw bit error at most 10% across four tuning crops × four independent fixed maps. |
| Real PNG recovery | 36/36 exact authenticated recoveries: four tuning covers × three sizes (512×512, 513×517, 1024×1024) × empty, Unicode, full-capacity messages. |
| Resume | Two real dense-model updates match one update → checkpoint/restart → one update, including BatchNorm, Adam and next random output. |
| Export independence | Fresh isolated encoder/decoder processes match eager outputs within `1e-5` absolute/relative tolerance; exact authenticated text on native Mac and Linux CPU. |
| Failure handling | Wrong passwords, damaged PNGs/packages, invalid shapes/types, incomplete evaluation, disk/write failures, incompatible checkpoints and interrupted work fail safely. |
| Budget | Time and experiment count persist; lock prevents concurrent spending; deadline leaves incomplete gates open. |
| Regression | Backend/frontend, strict types, lint/format, schemas, file lengths, locked containers and isolated browser restart pass. |

All tuning attempts are reported before any success filter. Quality measurements
come from saved/reopened integer PNGs: PSNR, SSIM and clipping, with elapsed time
and peak process memory. A small learning pass is not the release gate of 99.9%
recovery on 10,000 frozen full-capacity images with median 40 dB / 0.98 quality.

## Measured acceptance

- Native environment: macOS 26.6.2 arm64, Python 3.12.13, PyTorch 2.14.0,
  NumPy 2.5.3; four CPU threads. Linux: Python 3.12.14, PyTorch 2.14.0+cpu.
- One baseline experiment, `sprint04_baseline`, completed 1,000 steps in
  810.55 seconds (13 minutes 30.55 seconds). No diagnostic rerun or GPU was used.
- At acceptance, the persistent ledger recorded 1,354.87 seconds (22 minutes 34.87 seconds)
  across training, data checks, evaluation, export, and both package matrices.
  This used one experiment and stayed within its two-hour limit and the
  four-hour total. The ledger has no active reservation.
- The highest-ranked measured checkpoint was selected before the full proof:
  `checkpoint_1000_2a70b982465a4f42af7d78136f4c3cba`. The unmeasured latest
  recovery checkpoint was not used for model selection.
- All ten periodic validations recovered 4/4 messages. The selected checkpoint
  had 26.366 dB median PSNR and 0.85974 median SSIM on that four-case suite.

| Check | Recorded result |
|---|---|
| Backend regression | 351 tests pass; existing Starlette/httpx warning only |
| Frontend | 10 tests, TypeScript, ESLint, Prettier, and production build pass |
| Python quality | Ruff, formatting, strict mypy (including verification script), contracts and whitespace checks pass |
| File length | 148 source files; every file below 300 lines |
| Safe continuation | Real dense-model uninterrupted/resumed results match; real SIGTERM saves a completed step and restores signal handlers |
| Native full learning proof | 16/16 bit cases, raw BER 0.08865833; 36/36 exact authenticated PNG recoveries; gate passed |
| Native proof quality | Median PSNR 26.21893 dB; median SSIM 0.8650300; mean clipped fraction 0.02153964 |
| Native proof resources | 67.66 seconds; 10,113,499,136 bytes lifetime peak process memory (about 9.42 GiB) |
| Export tensor parity | Six fresh-process cases; maximum absolute eager/export difference 0.0 |
| Containers | Both locked CPU images build and become healthy |
| Browser | Isolated port 8084 save/reload/backend-restart/reset/accessibility flow passes |
| Native package text matrix | 36/36 exact recoveries; completed in 163.54 seconds |
| Linux package text matrix | 36/36 exact recoveries; completed in 264.86 seconds |

- Passed: every Sprint 4 acceptance gate listed above.
- Failed release targets: image quality is below 40 dB PSNR / 0.98 SSIM.
- Incomplete later gates: 10,000-image benchmark, held-out generalization,
  detection resistance, 4K, and the GPU pilot. These were outside this sprint.

These 36 trials use four tuning covers, not 36 independent source images. They
include full 1,024-byte messages at 1024×1024 and preserve Unicode, combining
characters, newlines, and NUL bytes exactly. No recovery attempts were filtered
or retried. The raw-bit suite uses 16 separate fixed random maps.

The quality targets for release are **not met**: 26.22 dB / 0.865 are below
40 dB / 0.98. The 10,000-image benchmark, held-out generalization, detection
resistance, 4K, and the GPU pilot remain unmeasured. Public integer-channel/oracle
unit fixtures test wrappers only and are not counted as learned-model evidence.

Model pair compatibility identity:
`dense_v1_0924c5b00889e44a6b327f90c8977f204cc7cb617b4bc6fd5c8848cc45a627ac`.
Selected state checksum:
`33b5cdb26008462d75986876a2ad6c413c76a330023c65aa2578df4467e70b16`.
Training source identity:
`9075627ef62d57601d82e5d821666cff07ec61468269674d9ced666004972b9e`.
During acceptance, input-type, preflight-budget, retry, and evaluation-error
guards were hardened; the learned architecture and optimizer math were unchanged.
The recorded checkpoint retains the exact source identity used for its run.
Evaluation and export can read that checkpoint, but this final checkout rejects
training resume from it because its source identity changed. Resuming it would
require its original source snapshot. The final checkout's deterministic resume
test passes with checkpoints created by that same checkout; the baseline has
already completed all 1,000 allowed steps.

Linux package checks used the local arm64 CPU image
`sha256:153bd0c17e6ce12ab203fd277d724e334e21b37d1b7eda77f6e1587084026649`.

Detailed local evidence:

- `logs/2026-09-23_21-41-42_sprint04_baseline_w33w877q/`: steps, validations and run summary.
- `logs/2026-09-23_21-57-17_sprint04_baseline_2hfmud6o/evaluation.json`: all 36 native attempts.
- `models/001V_2026-09-24/`: independent packages and tensor verification report.
- `logs/2026-09-23_22-01-32_sprint04_baseline_5hv1hp78/native_cpu_package_recovery.json`: all 36 independent Mac package trials.
- `logs/2026-09-23_22-07-30_sprint04_baseline_qp1ydhh2/linux_cpu_package_recovery.json`: all 36 independent Linux package trials.
- `state/cpu_proof/ledger.json`: persistent experiment accounting.

Model/data/log artifacts remain local and excluded from Git. The existing app
on port 8081 was not restarted or reset. Acceptance used the separate Compose
project `stegolab-sprint04-check` on port 8084; its containers, network, and volume
were removed after verification.

### Saved before/after viewing examples

- On 2026-09-24, four fresh 1024×1024 tuning-cover examples were saved with the
  same exported model and 1,024-byte public messages. All four recovered exactly.
  These are viewing examples, not retained PNGs from the original acceptance run.
- The local viewer is `.runtime/sprint04-comparison/index.html`; each `sample-N/`
  contains `before.png`, `after.png`, and an absolute-difference image amplified
  four times for visibility. `report.json` records quality and file checksums.
- Evidence: `logs/2026-09-24_04-37-20_sprint04_baseline_q1ynl4rp/viewing_examples.json`.
  The first invocation stopped during report serialization; its saved PNG was
  preserved and checked again without re-encoding or replacing randomness.
- Viewing generation added 45.31 seconds to the same ledger. Total consumption
  is now 1,400.18 seconds (23 minutes 20.18 seconds); no new training experiment.

## Repeatable commands

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest tests/backend
uv run --locked python -m backend_service.export_contracts --check
uv run --locked python scripts/check_file_lengths.py
uv run --locked stegolab verify_protocol
uv run --locked stegolab train docs/examples/cpu_proof_train.json
npm --prefix user_interface run lint
npm --prefix user_interface test
npm --prefix user_interface run build
```

Use the [model guide](../docs/model_training.md) for evaluate/export/resume
requests. Follow the [execution order](execution_order.md) for the later pilot.
Preserve existing owner planning files and the prepared source datasets.
