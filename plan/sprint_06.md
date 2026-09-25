# Sprint 6 — Experimental Encode and Decode

Status: delivered on stacked branches; local acceptance, CPU CI and container round trips pass.
Date: 2026-09-25. Planned duration: ten working days. Team: owner + Codex.
Branches: `codex/sprint-06-model-installation`, `-inference-uploads`, `-inference-jobs`,
`-encode-decode-interface`, `-critic-fork`, `-ledger-records`, `-acceptance-docs`.
Source baseline: `7bc7b55` (training-first GUI on `main`).

## Agreed outcome

- Make Encode and Decode work end to end in the browser with the Sprint 4 export
  `001V_2026-09-24` installed by explicit promotion, labelled experimental.
- Keep every password, message and recovered text out of databases, sidecars,
  logs, events and errors. Release only a PNG the matching decoder recovered.
- Build the critic and the full-state fork, and the ledger commands the Sprint 7
  decisions need, without running any GPU job or session. The only instance
  time charged is the 3,629 s host run of 2026-09-24, recorded below.
- No release-quality or SOTA claim. `pilot_ready` stays false.

## Delivered behavior

- `POST /models/install` verifies an export pair and its verification record and
  advertises it through capabilities; removal is refused while a job uses it.
- Raw-body uploads with a 16 MiB limit: covers become metadata-free PNG files,
  encoded uploads are kept byte for byte, both expire after 24 hours, and new
  files are refused while stored uploads and results would pass 512 MiB.
  `POST /capacity` reports the message limit for one image and model
  and refuses images outside the model's 512–1024 side range.
- Encode and decode jobs run on the existing single scheduler lane. The persisted
  record holds references, sizes and a byte count; the secrets live in memory
  until the job runs, keyed into the retry fingerprint through an in-memory HMAC
  key. The encoder runs in an isolated process with secrets on stdin; the
  matching decoder must recover the message before the PNG is published. Decoded
  text lives in memory for five minutes and is read with `no-store`.
- The Encode and Decode tabs: install button in Train, drop-zone upload with
  progress, model choice, live "X of Y bytes used" counter, password cleared on
  send, plain phase wording, side-by-side review, verified download, Copy and
  Clear, and a persistent experimental note.
- Critic (three-stage realism critic, weight-clipped, one encoder forward per
  batch; a zero weight is bit-identical to no critic) and `fork_from_checkpoint`
  (full state into a new experiment with recorded provenance), built and tested.
- Ledger `attest-closed`, `transfer`, per-stage totals and reason codes; the
  2026-09-24 hour is charged to setup.

## Measured locally

Machine: the owner's Mac, CPU only. Package: `001V_2026-09-24`, smooth
synthetic covers, three runs each, medians.

| Image | Message | Encode | Decode | Exact recoveries | Encoded PNG |
|---|---|---|---|---|---|
| 512 × 512 | 256 bytes | 1.87 s | 1.79 s | 3/3 | 462 KB |
| 1024 × 1024 | 1,000 bytes | 4.0 s | 2.48 s | 3/3 | 1.85 MB |

Through the CPU containers on the same machine (nginx proxy, isolated package
processes, encoder followed by decoder verification), the 1024 × 1024 encode job
took 12.7 s, 15.7 s and 18.3 s in three runs, decode of the re-uploaded file
7.6 s to 8.2 s, and a wrong password failed in 7.6 s to 11.9 s with the frozen
message. The backend container's memory peaked at 2.8 GiB of its 10 GiB limit,
sampled every few seconds during the run with the memory sampler adding load.
A 15.3 MB PNG uploaded through the proxy in 0.95 s; a 19 MB file was refused by
nginx with 413. A random-noise cover fails decoder verification in about 7 s and
publishes nothing; this is model behaviour, not a pipeline fault.

## Acceptance

### Passed locally

- Backend: 483 tests pass (3 min 24 s). Frontend: 31 unit tests, lint, type
  check and production build pass. Browser: five Playwright tests pass against
  the CPU containers with the backend restart enabled (34 s), including the
  fixture round trip: install from Train, encode a Unicode message, inspect the
  downloaded PNG bytes, decode the exact text, Copy, Clear (DELETE → 404), wrong
  password and never-encoded cover refused, axe on both tabs.
- Ruff lint/format, strict Python types, generated contracts and the file-length
  rule pass; all 300 checked source files contain fewer than 300 lines.
- Secret absence: a sentinel password and message never appear in
  `workflows.sqlite3`, `stegolab.sqlite3`, `installed_models.sqlite3`, any
  sidecar, log, event, job listing, process argument or environment.
- The real package round trip above; noise-cover verification failure; restart
  marks unfinished inference jobs interrupted with `inputs_lost`; cancel drops
  secrets; the proof lock no longer blocks inference or jobs queued behind a
  locked training job.
- Critic: weight zero leaves encoder, decoder, sampler and pair identifier
  bit-identical after two steps; a critic run resumes exactly through a
  version-2 checkpoint; fork equals resume and keeps provenance; refusals for
  the same experiment, another schedule, another dataset and a CPU-proof
  checkpoint; the CPU smoke command completes with the critic on.
- Ledger: the attested hour, chronological insertion, overlap and allocation
  refusals, transfers within the rules, closed sessions charged by used time,
  and operator mistakes named as mistakes. The operator ledger `state/gpu_pilot`
  now records the 2026-09-24 session: setup 3,629 s consumed, 3,571 s remaining;
  82,771 s of the 86,400 s allowance remain.
- Four adversarial review rounds during implementation (session notes, not
  stored in the repository) confirmed 37 defects before commit; all were fixed
  in the same branches, including a pre-existing shutdown bug where a refused
  second application rewrote the owner's live queue.

### Passed in CI

- Pull requests #10, #11, #12, #14, #15 and #16 merged with the CPU application
  check passing, and with the native CUDA image check passing where that
  workflow runs (it skips interface-only changes). #13, the Encode and Decode
  tabs, was merged into its feature branch while its CPU check was failing at
  the fixture-package step; the fix (commit 40873da) reached `main` green in
  #16. #17 (ledger) is merged into the critic branch and #18 carries the critic
  and ledger work to `main`. CI now runs on pull requests and `main` only,
  type-checks the tests, caches the browser, and builds the labelled fixture
  package before the containers start.
- Two CI-only failures were found and fixed during the sprint: the fixture
  builder needed a `models` folder that a fresh checkout lacks, and the browser
  restart test collided with the encode test on a second worker.

### Failed

- No remaining local or CI failures.

### Not run

- Anything on a GPU: the readiness drill, the baseline, the critic ablation,
  CUDA inference or CUDA package parity. No GPU job or session ran in this
  sprint; the 3,629 s of instance time from the 2026-09-24 host run, before the
  sprint's first commit, were attested and charged to setup.
- The real `001V_2026-09-24` package in CI: `models/` is not tracked, so CI
  uses the labelled integer-channel fixture, which is not a learned model.
- Images above 1024 pixels, tiling, JPEG output, 4K resource checks, detection
  resistance, and every release-quality gate. The Sprint 4 model stays at
  36/36 tuning recoveries at 26.22 dB / 0.865 SSIM, below release targets.

Review: pull requests #10 to #17 on `DanielGordon-ml/StegoLab`.

## Deviations recorded

- Interrupted inference jobs are marked `interrupted` with `inputs_lost` rather
  than `needs_input`; `needs_input` stays reserved for the resumable GPU path.
- The PRD's "No hidden payload detected or image is corrupted" is superseded by
  "No valid hidden message could be recovered. Check the password, model, and
  image."
- Replay of an inference request binds the secrets through an in-memory key, so
  a retry after a backend restart reports a conflict instead of replaying.
- The 500 GiB unencrypted root volume of `i-0ccaee67f0acaa574` is accepted;
  data, checkpoints and the ledger live on the encrypted 200 GiB volume.
- The planned `docs/encode_decode.md` guide became sections of the existing
  [GUI guide](../docs/gui_training.md), and no `sprint-06-contracts-frozen` tag
  was created: contracts changed slice by slice with the schema check as the
  guard.

## Remaining gates

- Sprint 7 opens the paid readiness drill inside the remaining 3,571 setup
  seconds (a retry may borrow at most 3,600 s from ablation with `transfer`),
  then the 14-hour baseline, with the host fixes listed in the
  [Sprint 6 draft](sprints/sprint6.md): hop limit 1, termination protection,
  EventBridge stop backstop, Budgets alert, unattended upgrades disabled.
- Before the baseline: parallel decoding in the sampler, checkpoint pinning, the
  near-duplicate audit as a separate report, a throughput check, and a separate
  infrastructure-passed result.
- The critic ablation only after the baseline has learned recovery, forked from
  the pinned checkpoint at matched update counts.
- Move the untracked root `config` file to `~/.ssh/config`; it is ignored now.

Guides: [GUI guide](../docs/gui_training.md), [pilot guide](../docs/pilot_training.md),
[GPU preparation](../docs/gpu_preparation.md), and [execution order](execution_order.md).
