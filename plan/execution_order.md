# StegoLab Execution Order

Status: researched delivery plan. Research date: 2026-09-23.
Related plans: [Backend](backend.md), [Frontend](frontend.md), [Infrastructure](infrastructure.md).

Delivery status: [Sprint 1](sprint_01.md) and [Sprint 2](sprint_02.md) are implemented
and locally verified. [Sprint 3](sprint_03.md) implements the local preparation and manifest slice of
Step 3. All 6,073 local UHD-IQA images, including 120 grayscale images, are
prepared and verified with their source splits. The acceptance record includes
sample and full-corpus measurements, plus offline container validation.
Remote adapters and full pilot data readiness remain open; Sprint 3 does not
complete the whole dataset step or replace the planned COCO benchmark.

[Sprint 4](sprint_04.md) implements the experimental CPU slice of Step 4:
model/quantization code, `train`, `evaluate`, `inspect_checkpoint`, and
`export_models`, full-state checkpoints, and persistent proof time accounting.
Use [the model guide](../docs/model_training.md) and Sprint 4's acceptance record
for the actual measured result. CLI delivery does not complete model-quality,
GPU, worker/API, or GUI gates. Public model operations remain unavailable and
usable payload capacity remains zero.

[Sprint 5](sprint_05.md) prepares the first GPU readiness test without launching
EC2: a versioned pilot CLI, lazy full-UHD loading, exact resume, explicit device
exports, CUDA image preparation, and separate instance-time budget controls.
CPU checks do not qualify GPU execution. The later readiness drill uses the
existing two-hour setup allocation. Full pilot data and release gates stay open.

## 1. Deliverables and shared decisions

- The initial planning deliverable was four Markdown files: backend.md, frontend.md, infrastructure.md, and execution_order.md under plan/. Sprint plans and acceptance records extend those plans as delivery progresses.
- Keep the existing CLAUDE.md and AGENT.md rules intact. Implementation follows their directory, naming, small-file, validation, and CI requirements.
- First release includes four GUI tabs, local/Hugging Face/HTTPS data, shared-password encryption, unchanged-PNG recovery, preserved dimensions, resumable training, and independent torch.export packages.
- Target 1,024 net UTF-8 bytes at 1024 × 1024. Other dimensions use tested capacity tiers defined by the backend.
- Development is on Mac CPU containers; the full pilot is on one AWS EC2 GPU with a 24 allocated-hour total budget.
- The research review is not a reproduced benchmark. Model quality and SOTA remain empirical gates, not assumed outcomes.
- Every feature uses a meaningful codex/ branch. Keep dependency order visible in pull requests, run relevant checks, and remove merged/deployed branches according to project rules.

## 2. Dependency order and gates

| Step | Work | Depends on | Required evidence before the next dependent step |
|---|---|---|---|
| 0. Reviewable plans | Save these four documents, cross-link them, record sources and assumptions, and check agreement across them. | Research | Four readable files, no broken internal links, no contradictory limits/states/export formats. |
| 1. Foundation and contracts | Create package structure, CPU Compose setup, configuration/schema modules, API skeleton, job records, CLI shell, and lean CI. Resolve frontend binding generation through its bounded proof/fallback. | 0 | Strict request/response validation, schema/client contract check, clean CPU startup, no committed code file reaches 300 lines. |
| 2. Message and image protocol | Implement UTF-8 framing, Argon2id/XChaCha encryption, error correction, capacity accounting, image preparation, PNG round trips, and known-answer fixtures. | 1 | Exact byte tests; wrong-password/corruption rejection; dimension and capacity tests; no secret persistence. |
| 3. Dataset adapters | Implement local upload/folders, Hugging Face inspection/download, HTTPS assets, cache, validation, and pause/cancel behavior. Freeze the pilot source manifests and splits. | 1 | Verified counts/checksums, version pinning, offline reuse, safe extraction, blocked unsafe URLs, cancellation cleanup. |
| 4. Model and training engine | Implement the exportable dense/residual baseline, quantization path, training/evaluation CLI, full checkpoints, and CPU export proof. | 2; dataset interface from 3 | Tensor contracts, tiny-set learning, separate export loading, safe stop/resume, saved-PNG evaluation. |
| 5. Four-tab frontend | Build the shared shell, Encode/Decode, dataset/Train flows, Config, snapshots/events, and reconnect/needs-input states. | Contracts from 1; integrate as 2–4 become available | Browser flows with clearly labelled test fixtures; shared validation; accessible controls; no canvas modification of downloadable PNG. |
| 6. EC2 readiness | Prepare CUDA target, manual host runbook, persistent storage, private access, budget ledger/stop guard, and recovery checks. GPU checks run within Step 7's setup allocation. | 1; export proof from 4 | Fresh GPU startup, CPU/GPU export smoke checks, restore test, short-budget stop drill, no public application port; all GPU time charged to the pilot. |
| 7. Capped research pilot | Run the fixed schedule below and save all evidence. | 2–4 and 6; frozen dataset/test manifests | Honest result report, checkpoints, exported candidates, runtime/memory measurements, and stopped EC2 session. |
| 8. Release integration | Replace test fixtures with selected real model packages; complete four-tab round trips, offline/cache flows, restart recovery, 4K limits, and documentation. | 5 and successful model gates from 7 | Full release acceptance on the intended hardware. Failing profiles stay experimental. |
| 9. SOTA research | Independently train detectors; compare matched classical/neural baselines; explore robust channels and higher capacities. | Reproducible first release and a separate compute budget | Named benchmark, fair baselines, independent tests, uncertainty, reproducible artifacts. |

- Parallel work after Step 1: protocol, dataset adapters, frontend shell, and infrastructure setup can proceed independently against the frozen contracts.
- Model training depends on protocol and data validation. Full GUI integration depends on real exports; a mocked page is not model acceptance evidence.
- Sprint 4 can use the completed local dataset interface while remote adapters
  remain open. Its bounded CPU proof uses a small frozen development sample and
  a separate two-experiment/four-hour ledger, with at most two hours per experiment
  including evaluation/export. It does not replace pilot data readiness or spend
  the 24-hour GPU allocation. Twenty minutes per remaining experiment allowance
  are reserved for saves and checks.
- Do not spend GPU-instance time waiting for UI work or manual reviews. Finish CPU/export/setup checks first.
- Step 6's GPU readiness and stop drill consume the two-hour setup allocation below. Step 8's final GPU integration checks consume the four-hour evaluation allocation; remaining CPU/browser checks can finish after shutdown. If readiness or integration needs more time, reduce training/ablation time within the same 24-hour ledger rather than add unbudgeted sessions.

## 3. Fixed 24-hour pilot

All allocations include their setup/checkpoint/shutdown overhead; the hard total covers every session and does not reset with a job.

| Maximum allocation | Experiment | Decision rule |
|---|---|---|
| 2 hours | GPU readiness, stop-guard drill, environment, data manifests, memory/throughput probe, tiny-set learning, PNG and export checks. | Stop with diagnostics if basic execution or quantized recovery is broken. |
| 14 hours | Train the initial dense/residual model, with validation and periodic recovery checkpoints. | Keep the best validation candidates and a checkpoint about four hours before the baseline ends. |
| 4 hours | One controlled critic ablation from that earlier checkpoint, keeping the remaining settings fixed. | Compare at matched update counts; record throughput differences. If baseline recovery has not learned, cancel the ablation and use this slot for diagnosis. |
| 4 hours | Frozen PNG tests, export parity, final GPU release integration, high-resolution checks, classical comparison where permitted and feasible, failure review, final save and shutdown. | Report incomplete comparisons explicitly when time or implementation terms prevent them. Never exceed the deadline or silently weaken targets. |

- Use the same frozen dataset identities and protocol for compared runs. Record code, environment, model, and manifest hashes.
- Do not treat the old and new literature's incompatible bpp definitions as one leaderboard. Report raw/network and net/user capacity separately.
- Large detector training, extra seeds, and robust-profile training are later research work with a separate budget.
- An experiment that fails its target still produces a useful diagnostic artifact. It does not qualify the model for release.

## 4. Release acceptance

- On 10,000 frozen source images prepared at 1024 × 1024, encode a full 1,024-byte user message per image before encryption. Achieve at least 99.9% exact authenticated message recovery, median PSNR at least 40 dB, and median SSIM at least 0.98. Report confidence intervals, failures, refusals, retries, and preprocessing. Short/empty-message tests are separate and do not contribute to this gate.
- Freeze thresholds and source-image identities before viewing final test results. Report native high-resolution results separately from resized COCO results.
- Test messages at zero, short, and maximum capacity; English, Hebrew, emoji, combining characters, and line breaks; reject overflow.
- Verify the saved/reopened PNG before download, while reporting benchmark performance before that filter.
- Independently run exported encoder and decoder packages. The decoder needs neither the original cover nor the encoder; the encoder package does not depend on decoder weights.
- Pass image-size, odd-dimension, orientation/color preparation, tiling, and 4K resource checks. Measure latency rather than promising 'instant'.
- Pass local/HF/HTTPS dataset workflows, gated-access errors, offline cache reuse, safe cancel/cleanup, disk-full handling, and restart recovery.
- Pass checkpoint stop/resume and compatibility checks. Keep latest recovery, three best, and pinned checkpoints without deleting deployment exports.
- Pass browser reconnect/reload, needs-input recovery, configuration import/export/reset, secret handling, and keyboard/accessibility checks.
- Keep routine CI small. Run expensive GPU/benchmark checks at model-promotion/release gates, not on every documentation or UI change.

## 5. Research and maintenance rules

- A first-release pass proves only the tested PNG workflow. Detection resistance needs independent steganalysis; damaged-image recovery needs a separate channel profile.
- Treat LISO, newer binary methods, image-in-image systems, and robust watermarking as different comparison categories. The linked backend evidence table records why each is included or deferred.
- Keep unverified claims labelled. Missing full text, code, weights, license, or reproduction means a method cannot be declared the winner.
- Dataset revisions and downloaded counts are part of the experiment. Never silently replace an unavailable mirror, split, image preparation, or model.
- Preserve a known-good application image/model pair for rollback. New model promotion is explicit and does not overwrite the only working export.
- Create the interactive Archify architecture map during implementation and keep it aligned with the two-container deployment, worker processes, persistent storage, and independent inference packages.
- No open product-choice questions remain for this first-release plan. Cloud region/access inputs are supplied at deployment, and scientific outcomes are resolved by the bounded experiments and gates above.
