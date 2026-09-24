# Sprint 5 — Prepare the First GPU Run

Status: implementation and acceptance checks in progress.
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

Final test counts, container/browser checks and CI outcomes will be recorded here
after integration. Actual GPU training, CUDA resume/recovery/parity and physical
shutdown remain `not_run`, regardless of passing simulated or CPU tests.

## Remaining gates

- The later paid readiness drill consumes the existing two-hour setup allocation.
- Near-duplicate review and remaining experiment support precede the full pilot.
- COCO release data, critic work, 4K, GUI integration, detection resistance and
  model-quality targets remain open. Dataset `pilot_ready` remains false.
- User-owned planning drafts and the prepared source datasets are preserved.

Guides: [pilot commands](../docs/pilot_training.md),
[data selection](../docs/pilot_data.md), [GPU preparation](../docs/gpu_preparation.md),
and [execution order](execution_order.md).
