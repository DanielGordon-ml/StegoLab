# Sprint 7 — Dataset Adapters

## Status and goal

- **Sprints 1–6 are complete and merged** (`main` at `0063e00`). Fresh checks from the Sprint 6
  record: 483 backend tests, 31 frontend tests, 5 browser tests, all source files below 300 lines.
- **Sprint 7 goal:** a user can bring a dataset into StegoLab from a Hugging Face repository, an
  HTTPS archive or a browser upload, watch bytes arrive, pause or cancel, and get a prepared
  revision under `datasets/` with the raw files saved under `data/<source_name>/` like the
  UHD-IQA folder. The frozen release-benchmark identities and the near-duplicate audit report
  exist and preflight reads the report. No GPU second is used and `pilot_ready` stays false.
- **Duration:** 10 working days. Team: you + Codex, with a parallel agent for the browser work.
- Confirmed scope (owner decisions, 2026-09-26): the GPU readiness drill and the 14-hour baseline
  move to Sprint 8 unchanged; the archive upload is a separate 2 GiB chunked route; the Hugging
  Face token lives only in the backend container; raw data lives under `data/<source_name>/`.
- No release-quality or SOTA claim. Downloading and preparing a dataset never makes it
  "pilot-ready" or "compatible"; every summary says so.

## Backlog

| Days | Work | Completion evidence |
|---|---|---|
| 1–2 | Contracts first: source specifications, fetch request/result/progress, cache, upload session, storage summary, benchmark identities, audit report and preflight fields, capability fields, the widened `source_kind`; regenerated `contracts/`. Foundation: disk reserve helper, `DatasetWriter.write_stream`, address policy, secure transport, transfer with Range resume, hidden-entry scan rule, compose mounts for `data/` and `.cache/stegolab`, setup script, `pyarrow` lock. Freeze contracts at the end of day 2 so the browser agent starts. | Contract check passes; transport tests prove blocked private/loopback/metadata addresses, redirect re-validation, token sent only to `huggingface.co`, resume with `If-Range`; the existing 483 backend tests pass and the full UHD-IQA revision still loads (no manifest field added). |
| 3–4 | Cache with verified offline reuse and unused-entry removal; safe zip/tar extraction; materialization into `data/<source_name>/` with the provenance marker and split CSV; HTTPS-archive adapter; Hugging Face adapter over the Hub REST API with LFS sha256 checks, gated guidance and the parquet reader; upload session store and upload adapter; `stegolab fetch_dataset` and `inspect_source`. | A DIV2K-shaped zip fixture ends as `data/<name>/` with marker and CSV and then a prepared revision whose manifest carries provenance and `split_mapping`; traversal, symlink, nested-archive, bomb and lying-size fixtures are rejected; a gated 403 gives the fixed message; no token text in any log or event; a second run downloads zero bytes; SIGTERM keeps the partial and cleans staging. |
| 5–6 | Job integration: fetch job record in the store union, runner with the progress sidecar, per-command time limit and stall rule, cancel-while-running, pause, resume and cancel-while-paused; `/api/v1/datasets` routes, upload routes, capabilities and workspace folders, nginx block. Browser agent from the frozen contracts: source selector, four forms, inspect step, chunked resumable upload, transfer progress, summary card, storage card. | Route tests and fake-process job tests pass; the one real subprocess round trip (upload session → fetch job → `data/<name>/` → revision visible in `GET /workspace`) passes; frontend unit tests pass; a manual browser fetch of the wikitext parquet on port 8081 shows bytes and completes. |
| 7–8 | Research data: benchmark identities schema, `freeze_benchmark` and `validate_benchmark`; perceptual hash, `audit_near_duplicates` and the `.audits/` report; preflight reader. Playwright test for the upload path and the cancel/pause display; accessibility at 1280 and 390 px; docs (`docs/datasets.md`, GUI guide, local setup, pilot guide); Archify map update. | Synthetic tests pass (identical 0, resized and recompressed ≤ 4, unrelated ≥ 20, crop documented as missed); `--limit 200` then the full audit of the UHD-IQA revision writes a report in 10–15 minutes and preflight turns the check into passed or failed; CI green including the browser test. |
| 9–10 | Acceptance runs on the Mac: wikitext (pinned parquet, cache reuse), COCO annotations + `val2017` from the pinned mirror → freeze and commit the identities → evaluation-only revision → audits → preflight; DIV2K validation HR over HTTPS with pause and resume; gated ImageNet without a token (guidance, zero bytes) and with the owner token (one bounded shard). Doc drift, `plan/sprint_07.md` record with measured numbers, pull requests, branch cleanup, Sprint 8 draft pointer. | The identities file is committed before any benchmark pixel is prepared; the record lists sizes, sha256s, elapsed times, counts by split, audit tables and preflight statuses honestly; `pilot_ready` stays false; GPU time used is zero. |

- Codex owns the backend, tests and integration. A parallel agent owns the browser work after the
  contracts freeze. A separate review checks address handling, extraction limits and secret absence.
- Use feature branches with the `codex/sprint-07-` prefix (`dataset-contracts`,
  `dataset-transport`, `dataset-sources`, `dataset-jobs-routes`, `dataset-interface`,
  `benchmark-audit`, `acceptance-docs`). Keep source files below 300 lines.
- You review the contracts on day 2 and the working demo at sprint end.

## Implementation decisions

- **Raw data location:** every remote or uploaded source is materialized at `data/<source_name>/`
  (extracted members with their archive paths, parquet images as `<split>/<sha256>.<ext>`, text
  corpora as `.txt`), with a hidden `.stegolab_source.json` marker (kind, reference, resolved
  revision, asset sha256s, member count, split mapping, completed) and a `source-metadata.csv`
  in the UHD-IQA format when splits are declared. Publication is atomic through the existing
  `DatasetWriter`; an existing folder is reused only when the marker matches, never overwritten.
  `data/` is registered as a server folder, so "Server folder" preparation of a materialized
  folder produces a labelled manifest with provenance instead of an unlabelled `local` one.
- **Cache:** `.cache/stegolab/datasets/<kind>/<asset identity>/` holds only immutable,
  sha256-verified archive and parquet bytes plus `.partial` folders; completed entries are shared
  across source names and reused offline after re-hashing; cancel removes only the job's partials;
  "Remove unused downloads" deletes unreferenced entries and is refused while a fetch job exists.
- **Manifest constraint:** `DatasetManifest` gains no field (its loader needs byte-exact canonical
  JSON, so a new field would break the 57 GB revision). Provenance goes into `source_provenance`
  strings; the existing `split_mapping` holds the per-source mapping; `source_kind` widens to
  `hugging_face`, `https_archive`, `upload`; `training_intended: false` on the request allows
  evaluation-only revisions (COCO validation → `held_out`).
- **Transport:** stdlib only. Resolve once, check every address (IPv4 and IPv6, unwrapped
  mapped forms; reject loopback, link-local and metadata, private, CGNAT, multicast, reserved),
  connect to that exact address with TLS verification on the URL host, follow at most five
  redirects with the same checks per hop, attach the token only to `huggingface.co`, bound
  connect/read/stall time and bytes, never consult proxy variables, never log a URL query string.
- **Hugging Face:** Hub REST only, no `huggingface_hub` or `datasets` library, no dataset scripts.
  Revisions resolve to a commit sha that the manifest pins; LFS sha256 and sizes come from the
  tree API; file selection by prefix, names, split shard pattern and allow-list; gated or missing
  access gives one fixed message and downloads nothing. Parquet needs `pyarrow`, the only new
  runtime dependency, read row group by row group.
- **HTTPS archive:** HTTPS only, no way to disable certificate checks (the official COCO host fails
  verification, so COCO comes from the pinned Hub mirror); ETag plus `Accept-Ranges` decide
  whether pause is offered; optional expected sha256; `archive_splits` maps member folders to
  splits.
- **Upload:** a separate session store on the data volume; 16 MiB chunks, at most 128, any order,
  resumable after reload, whole-file sha256 computed on completion; the archive becomes a cache
  asset and then an ordinary fetch job; no pause.
- **Safe extraction:** streamed zip and tar into staging on the data volume; validated relative
  paths; extension allow-list; magic sniff; regular files only; nested archives, links, devices,
  encrypted entries, duplicates rejected; per-member, total and count limits; actual bytes checked
  against declared sizes; everything else recorded as a rejection.
- **Jobs, progress, cancel, pause:** fetch jobs run on the existing single lane as a child
  process that writes a progress sidecar; the snapshot shows bytes only when the total is known;
  cancel of a running job shows "Cancelling" until the worker exits and the parent removes the
  job's partials; pause is offered only for resumable sources and resumes from the cached bytes
  with a Range request; the worker never deletes partials itself; restart leaves paused jobs
  paused. Fetch jobs get a 24-hour limit and a 30-minute no-progress rule.
- **Routes and browser:** inspection before fetch is synchronous with a 20-second deadline;
  `GET /workspace` stays the single read model and lists `data/` subfolders; one fetch job at a
  time; the Train tab offers four sources with an inspect step, byte progress, Pause/Resume/Cancel
  from the backend's actions, an honest summary card, and a Config storage card. Wording stays
  B1/B2 and never says "compatible" or "pilot-ready" for a fetched dataset.
- **Token:** read from a secret file or `HF_TOKEN` in the backend container; capabilities expose
  only `hugging_face_token_configured`; never in logs, events, exports or the browser.
- **Benchmark identities:** frozen in git at `docs/benchmarks/release_benchmark_v1/` from the
  COCO annotations and the mirror's LFS sha256 values: all 5,000 `val2017` images plus 5,000
  training-source and 1,000 reserve members chosen by a seeded hash order over `train2017`;
  `identities_checksum` is the reference everywhere; no pixel or byte hash of `train2017` is
  needed now; the 1024 preparation policy is a later frozen policy.
- **Near-duplicate audit:** a numpy-and-Pillow perceptual hash over prepared revisions, within
  splits and against the benchmark when its pixels exist, reported at distances 0, 4, 8 and 12
  without declaring one truth (8 is the automated gate), written outside the immutable revision
  under `datasets/.audits/`, bound by revision, records checksum, identities checksum and its own
  checksum, `pilot_ready: false` inside.
- **Preflight:** reads the report when the request names it; missing gives `not_run` with the
  command to run; wrong revision, limited report or cross-split pairs at 8 or less give `failed`;
  `release_benchmark` stays `not_run`; the flag stays false by construction.
- **Tests and CI:** no real network anywhere: loopback HTTP server behind a connector seam, a
  test-only address policy, one self-signed TLS test, recorded Hub JSON, hand-built archives,
  fake processes for job transitions, one real subprocess round trip, one Playwright spec with a
  real tiny zip. Expected CI growth about one minute. Existing large test files are not grown.
- **Repository hygiene:** work in a worktree; never commit the owner's private notes, `AGENTS.md`
  or `config`; `data/`, `.cache/`, `datasets/`, `secrets/` stay ignored; the 1.5 MB identities
  file is the one recorded exception to file-size norms.

## Definition of done

- Hugging Face, HTTPS-archive and upload sources fetch into `data/<source_name>/` and prepare
  into `datasets/`, with pinned revisions and provenance in the manifest, proven by tests and by
  the recorded local runs of wikitext, COCO validation, DIV2K validation and gated ImageNet.
- Blocked addresses, redirect checks, safe extraction, cancel cleanup, pause and resume, offline
  cache reuse and disk-full handling are covered by tests without network access.
- The identities file is committed, the audit report exists for the UHD-IQA revision, and
  preflight reads it. `pilot_ready` stays false everywhere.
- Contracts are regenerated, the schema check passes, all source files stay below 300 lines, CI is
  green with one new browser test.
- No token, URL query string or private path appears in any log, event, error or export.
- Documentation is updated in plain language; the Sprint 7 record lists what was measured and
  what remains open. GPU time used: zero. No release-quality or SOTA claim.

## Not in this sprint

Any GPU session, the readiness drill, the baseline; the 1024 × 1024 benchmark preparation policy
and benchmark evaluation; preparing `train2017` (fetching it is stretch only); a second worker
lane; image–text pairing in training; Flickr2K, BookCorpus, Visual Genome, Tiny-ImageNet, Common
Crawl; plain HTTP or unverified TLS sources; automatic removal of near duplicates; a Config token
field; an audit read route; a validation-job route; uploads above 2 GiB.

## Deferred to Sprint 8 — decisions already taken

- Everything the Sprint 6 draft parked for Sprint 7 moves unchanged: keep `i-0ccaee67f0acaa574`
  with hop limit 1, termination protection, EventBridge stop backstop, Budgets alert, unattended
  upgrades off; the lost hour is charged to setup (3,571 s remain for the drill, a retry may borrow
  at most 3,600 s from ablation); data transfer by a transfer instance and `rsync` (S3 fallback),
  now including `data/` and `.cache/stegolab`; CUDA image built once on the host.
- Before the baseline: parallel decoding in the sampler, checkpoint pinning, throughput check,
  the separate "infrastructure passed" result, the benchmark preparation policy with declared
  resizing, per-member byte hashes of `train2017`, and the exclusion list from the audit.
- The critic ablation only after the baseline has learned recovery.
