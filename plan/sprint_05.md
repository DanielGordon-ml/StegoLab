# Sprint 5 — Prepare the First GPU Run

Status: preparation implemented; local acceptance and native CUDA-image checks pass.
Date: 2026-09-24. Planned duration: ten working days. Team: owner + Codex.
Branch: `codex/sprint-05-gpu-preparation`. Source baseline: `0bcc16a`.

## Agreed outcome

- Prepare CPU-tested code, CUDA containers, cloud runbooks, and budget controls
  for a later GPU readiness test. No EC2 launch or GPU spending in this sprint.
- Use the existing UHD-IQA revision without changing images or splits.
- Preserve the Sprint 4 CPU proof and its separate ledger. Keep public model
  operations unavailable and usable capacity zero.
- Do not claim new model quality, release acceptance, or SOTA. Sprint 4 remains
  36/36 tuning recoveries at 26.22 dB / 0.865 SSIM, below release quality targets.

## Delivered behavior

- Schema-version-two pilot requests share `train`, `evaluate`, and `export_models`.
  `preflight_pilot` reports metadata, versions, storage, budget and pending gates.
- Lazy synchronous training reads frozen representatives and verifies image bytes
  before cropping. Resume preserves consumed samples and crop randomness.
- Dense FP32 training supports batches four/eight accumulated to sixteen, Adam
  `1e-4`, a frozen step count and a twenty-percent image-loss ramp to 100.
- Full checkpoints preserve networks, BatchNorm, optimizer, random states,
  sampler and code/data/numerical identities. Missing CUDA fails explicitly.
- GPU tuning evaluation retains all 32 full-message saved-PNG attempts. CPU smoke
  is bounded, unranked engineering validation, not a quality experiment.
- Independent version-two packages contain explicit CPU/CUDA dependency profiles
  and device selection. Existing CPU proof requests and packages remain supported.
- Separate native-amd64 CUDA image and CI smoke; no Apple Silicon emulation.
- A persistent 24-hour ledger accounts for instance startup through confirmed
  stop. Operation locks, deadline guards and external-stop templates preserve the
  original two/fourteen/four/four-hour stage allocations.

## Frozen data and local evidence

- Full revision: `75bef6dab2b3de136d1c5c359d6753a25e4a750d5387fe75921e394482f0776c`.
- Splits remain 4,269 train / 904 tuning / 900 held-out. No held-out scoring.
- Select 32 tuning covers from 902 with native 1024-pixel sides. Record the two
  3840×960 exclusions (`uhd_iqa:3125.jpg`, `uhd_iqa:774.jpg`).
- Selection checksum:
  `9460c45da3dc3e9c5c45170b4c288a135fb470fc045a8cf340c497cbe668104d`.
- Preflight passes locally; remaining GPU allowance is 86,400 seconds with no
  initialized session. Evidence: `logs/sprint05_acceptance/preflight.json`.
- A bounded loading probe read sixteen distinct training images into four batches
  in 2.125 seconds, with 363.2 MiB peak process memory. Prepared files were
  unchanged. No model, optimizer, tuning or held-out pixels were used.

## Acceptance

### Passed locally

- All 411 backend tests pass, including legacy CPU-proof behavior. One existing
  Starlette/httpx deprecation warning remains.
- All ten frontend tests, lint, type checks and production build pass.
- Ruff lint/format, strict Python types, generated contracts and four protocol
  fixtures pass. All 193 checked source files contain fewer than 300 lines.
- Real dense CPU runs match uninterrupted execution after resume for each fixed
  physical batch (four and eight): weights, BatchNorm, Adam, next random values,
  consumed sample positions, image order and crops. Each update uses sixteen images.
- Failure tests cover damaged data/checkpoints, incompatible state, missing CUDA,
  disk and interrupted writes, deadlines, retained exports, concurrent operations,
  clock regression, restart accounting, and failed shutdown commands.
- Independent CPU exports match eager execution, including existing protocol and
  odd-size cases. Requested CUDA publication is tested with simulated device
  verification; actual CUDA parity is still pending.
- CPU containers build and become healthy. The isolated Chromium test passes
  after reload and backend restart (one test, 2.1 seconds). Its temporary project
  and volumes were removed; the existing application on port 8081 was preserved.
- The documented full-revision CPU command completed two unranked optimizer
  updates in 7.760 seconds. Evidence:
  `.runtime/pilot_smoke/logs/2026-09-24_05-27-38_pilot_cpu_smoke_auen4b3a/training_run.json`.
  This is an engineering smoke test, not a new model-quality result.
- Readiness check-only passes metadata validation and records all five GPU checks
  as `not_run`. Explicit execution rejects non-setup budget sessions before any
  CUDA call, keeping the later drill within the existing two-hour setup allocation.

### Passed on native Linux amd64 CPU CI

- The locked PyTorch `2.14.0+cu126` image builds and runs without a GPU.
- Runtime dependency checks, CLI startup, real CPU forward/backward, saved-state
  continuation, graph save/reload, and unavailable-CUDA rejection pass.
- [Native build and smoke evidence](https://github.com/DanielGordon-ml/StegoLab/actions/runs/35960252970)
  covers implementation commit `407751b`. The image is 6,826,719,106 bytes;
  80 GiB of runner storage remains after the check.
- The full application CI also runs backend/frontend, CPU containers and browser
  restart checks. Its latest result is visible in
  [PR #7](https://github.com/DanielGordon-ml/StegoLab/pull/7/checks).

### Failed

- No remaining local acceptance failures or native CUDA-image smoke failures.
- Real GPU behavior is unmeasured; an unrun check is never counted as a pass.

### GPU and cloud checks not run

- CUDA hardware execution, device memory/throughput, batch selection and planned
  baseline step count; actual CUDA continuation, independent package parity and
  all-32 learned-message recovery.
- EC2 launch, physical host stop, external AWS stop backstop, persistent-volume
  restore and SSH access on a real cloud host. Local simulations and reviewed
  runbooks do not establish these results.
- GPU time used: **zero hours**. No GPU budget session was opened; the separate
  24-hour allowance and Sprint 4 proof ledger remain unchanged.

Review: [draft PR #7](https://github.com/DanielGordon-ml/StegoLab/pull/7).

## Remaining gates

- The later paid readiness drill consumes the existing two-hour setup allocation.
- Near-duplicate review and remaining experiment support precede the full pilot.
- COCO release data, critic work, 4K, GUI integration, detection resistance and
  model-quality targets remain open. Dataset `pilot_ready` remains false.
- User-owned planning drafts and the prepared source datasets are preserved.

Guides: [pilot commands](../docs/pilot_training.md),
[data selection](../docs/pilot_data.md), [GPU preparation](../docs/gpu_preparation.md),
and [execution order](execution_order.md).
