# Sprint 6 — Experimental Encode and Decode

## Status and goal

- **Sprints 1–5 and the training-first GUI (PR #9) are complete and merged** (`main` at `7bc7b55`).
  Fresh checks: 440 backend test cases (2 skipped), 22 frontend tests, 3 browser tests, 17 API routes,
  all source files below 300 lines.
- **Sprint 6 goal:** a non-technical user can hide a message in an image and recover it from the browser,
  using the existing Sprint 4 export as a clearly labelled **experimental** model. No message or password is
  ever saved to disk or logs.
- **Duration:** 10 working days. Team: you + Codex, with a parallel agent for the browser work.
- Confirmed scope (owner decision, 2026-09-25): Encode/Decode only. The critic and the fork-from-checkpoint
  code are built and tested but not run. No paid GPU session this sprint; the readiness drill and the 14-hour
  baseline move to Sprint 7 with the decisions recorded at the end of this file.
- The model is not release quality (median 26.2 dB PSNR / 0.865 SSIM against targets of 40 dB / 0.98 /
  99.9 % recovery on 10,000 images). Every screen says so. No SOTA claim.

## Backlog

| Days | Work | Completion evidence |
|---|---|---|
| 1–2 | Contracts first: model and inference schemas, rewritten capabilities, regenerated `contracts/`; check the 001V package still verifies under current dependency pins; installed-model store and `/models` routes; image uploads, capacity, image artifacts; nginx upload limit; fixture package script; ledger "attest closed session" and "transfer" commands; move the SSH `config` out of the repo; CI trigger, browser cache and type-check fixes. Freeze contracts at the end of day 2 (tag `sprint-06-contracts-frozen`). | Contract check passes; installing the fixture pair and `001V_2026-09-24` turns encoding and decoding on in a temporary workspace; a 1024×1024 JPEG upload returns a prepared summary with a 1,024-byte limit; a 1100×600 image is refused with the range message; ledger tests pass; `git status` shows no `config`. |
| 3–4 | Secret-safe encode and decode jobs: in-memory secret and decoded-text stores, isolated package runner with a process handle, admission checks, job runner (encode → verify with the decoder → publish PNG; decode → text in memory), routes, store request union, scheduler tweaks, restart handling. Browser agent starts from the frozen contracts: types, upload adapter, image picker, model choice. | A real round trip through the job service passes with the fixture package (encode → valid PNG → decode the exact Unicode text); wrong password gives the frozen message; a sentinel password and message never appear in any database, sidecar, log or event; cancel drops secrets; restart marks the job interrupted with "inputs lost"; test time grows by at most 35 seconds. |
| 5–6 | Encode and Decode panels, status component, experimental banner, Copy and Clear, active-job routing, frontend tests. Critic module, fork-from-checkpoint and their tests (build only). | Frontend tests pass; a manual browser round trip with 001V on port 8081 works; `test_pilot_critic.py` passes; a critic with weight zero is bit-identical to no critic. |
| 7–8 | One browser test (`encode_decode.spec.ts`) with the fixture package built in CI; accessibility checks on both tabs; `docs/encode_decode.md`; wording review. | CI is green including the browser round trip and PNG byte inspection; axe passes at 1280 and 390 px. |
| 9–10 | Doc drift (README, Architecture, execution order, Sprint 5 gates), `plan/sprint_06.md` acceptance record with measured numbers, pull requests, branch cleanup, Sprint 7 draft pointer. | Pull requests merged; the record lists passed, failed and not-run items honestly; `pilot_ready` stays false; GPU time used is zero. |

- Codex owns the backend, tests and integration. A parallel agent owns the browser work after the contracts freeze.
- Use feature branches with the `codex/sprint-06-` prefix (`model-installation`, `inference-jobs`,
  `encode-decode-interface`, `critic-fork`, `ledger-records`, `acceptance-docs`). Keep source files below 300 lines.
- You review the contracts on day 2 and the working demo at sprint end.

## Implementation decisions

- **Model installation:** a dedicated `installed_models.sqlite3` store, not a workspace registry kind. Install
  verifies both packages, equal compatibility identity, and the pair's `verification.json`; it copies nothing and
  does not re-run the known-answer check. Only an explicit `POST /models/install` installs a model; exports never
  install themselves; `DELETE /models/{id}` is refused while a job uses the model. HTTP install only; no CLI
  wrapper this sprint.
- **Capabilities:** replace the "always false" pins with real fields (installed models, profile, payload limit,
  image side range, 16 MiB upload limit) and a validator that keeps every field consistent with the installed
  list. `experimental_models_only` stays true by construction. Regenerate contracts and frontend fixtures in
  the same commit.
- **Uploads:** raw file body (no multipart dependency), 16 MiB limit because both containers keep small
  temporary filesystems, streamed to the application volume. A `purpose` of `cover` prepares the image;
  `encoded` stores the PNG unchanged for decoding. Uploads and results expire after 24 hours or a 512 MiB cap.
- **Image range:** this experimental model accepts sides between 512 and 1024 pixels. The range is checked at
  capacity and at job admission with a plain message, before any model process starts.
- **Encode and decode jobs:** same queue and job records as training, but a separate persisted record that holds
  only references and byte counts. The message and password live in a bounded in-memory store (8 entries) and
  reach the package runtime on stdin through an isolated `python -I -B` process with a 120-second limit and a
  process handle for safe shutdown. Encode runs the encoder, then the decoder on the saved PNG; only an exact
  match publishes the download. Decode keeps the text in memory for five minutes with a `no-store` endpoint;
  Clear deletes it.
- **Restart:** queued or running inference jobs become `interrupted` with the message "The application
  restarted before this job finished. The message and password were not saved. Start the job again."
  `needs_input` stays reserved for later resumable work (a recorded deviation from the backend plan).
- **Scheduler:** inference jobs are not blocked by the CPU proof lock; a 60-second maintenance tick sweeps
  expired uploads and texts. One job at a time keeps memory inside the 10 GB container limit; the browser says
  "Queued behind a training job" when that happens.
- **Errors and wording:** decode failure uses the frozen protocol text "No valid hidden message could be
  recovered. Check the password, model, and image." (the PRD wording is recorded as superseded). Encode
  verification failure: "The saved image did not pass message recovery with the matching decoder. Nothing was
  published. Try another cover image." Expired text: "The recovered text is no longer available. Enter the
  password and decode again."
- **Browser:** split the placeholder into encode panel, decode panel, image picker (drag and drop, object URL
  preview, upload progress), capacity counter ("X of Y bytes used" by UTF-8 bytes), status component, model
  choice, and an install button in the Train tab's model exports card. The download is a link to the exact
  verified PNG bytes, never a canvas copy. A persistent note on both tabs: "Experimental model. Encoded images
  are visibly reduced in quality and message recovery is not guaranteed. This is not release quality. Keep the
  PNG unchanged when sharing." Passwords clear after submission; decoded text clears on Clear and when leaving
  Decode; the active-job indicator opens the job's own tab.
- **Tests and CI:** the existing integer-channel test modules become a shared fixture and a script builds the
  same labelled pair into `models/fixture_not_a_neural_model/` for CI and Compose. Backend tests cover
  installation, uploads, capacity, the real round trip, secret absence, restart, expiry and route contracts.
  One new browser test does upload → encode → download → inspect PNG bytes → decode exact text, wrong password,
  and the no-model state. The real 001V run is local acceptance only, because `models/` is not in Git.
  Expected CI growth is about 1.5 minutes. Cheap lean-ups: pull-request-only triggers, cached browser
  download, type checks that include tests.
- **Critic and fork (build only):** a separate `model_critic.py` (SteganoGAN critic pattern, weight-clipped
  loss, one encoder forward per micro-batch), a `critic` configuration block that is off by default,
  checkpoint state version 2 with optional auxiliary state (version 1 still loads), and
  `fork_from_checkpoint` that copies the full training state into a new experiment with provenance.
  `model_networks.py` is not touched. No GPU checkpoint exists yet, so changing the pilot code identity now
  costs nothing.
- **Ledger:** add `attest-closed` (a past session with start and stop times and evidence) and `transfer`
  (move seconds between stages with a reason), print failure reasons, and show per-stage totals. Use them to
  record the 2026-09-24 hour (epochs 1790283913 → 1790287542) against the setup stage.
- **Repository hygiene:** move the untracked root `config` (an SSH client file with a public address) to
  `~/.ssh/config`, ignore `/config`, and delete the stale local `codex/sprint-01-*`, `codex/sprint-03-plan` and
  `codex/full-system-architecture` branches.

## Definition of done

- Encode and Decode work end to end in the browser with an installed experimental model, with the round trip
  proven by tests and by the recorded local run with 001V.
- No password, message or decoded text appears in any database, sidecar, log, event or error.
- Contracts are regenerated, the schema check passes, all source files stay below 300 lines, CI is green.
- Critic, fork and ledger commands exist with passing tests, but no training run used them.
- Documentation is updated in plain language, including the new user guide and the corrected architecture
  notes. The Sprint 6 record lists what was measured and what remains open.
- GPU time used: zero. No release-quality or SOTA claim. `pilot_ready` stays false.

## Not in this sprint

Images above 1024 pixels or tiling, GPU inference, JPEG output, automatic model promotion, pause or
"needs input" flows, cancelling a running inference job, uploads above 16 MiB, a second worker lane, a Config
model selector, remote dataset downloads, LPIPS, attack layers, mixed precision, any paid GPU session.

## Deferred to Sprint 7 — decisions already taken

- **The host:** g6.2xlarge `i-0ccaee67f0acaa574` in eu-central-1c exists and is stopped. Keep it. Before the
  first paid session: metadata hop limit 1, termination protection on (never stop protection), EventBridge
  stop backstop with a dry run, an AWS Budgets alert, unattended upgrades disabled. Record the 500 GiB
  unencrypted root as an accepted deviation (all data, checkpoints and the ledger live on the encrypted
  200 GiB volume). Storage costs about $2 per day while stopped.
- **The lost hour:** the host ran 3,629 seconds on 2026-09-24 with no ledger session. Charge it to the setup
  allocation with `attest-closed`; 3,571 seconds remain for the readiness drill. A drill retry may borrow at most
  3,600 seconds from the ablation stage with a recorded reason.
- **Data transfer:** the prepared dataset is 57 GB. Attach the data volume to a small transfer instance and
  `rsync` from the Mac (zero ledger time); S3 is the fallback. Build the CUDA image on the host once and record
  its identifier.
- **Before the baseline:** parallel decoding inside each batch in the sampler, automatic checkpoint pinning,
  the near-duplicate audit as a separate report read by preflight (the `pilot_ready` flag is part of a frozen
  checksum and stays false), a throughput check in the readiness script, and a separate "infrastructure passed"
  result so a recovery miss by the Sprint 4 model does not read as an infrastructure failure.
- **Critic ablation:** only after the baseline has learned recovery, forked from the pinned checkpoint,
  compared at matched update counts.
