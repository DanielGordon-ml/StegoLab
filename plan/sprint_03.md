# Sprint 3 — Local UHD-IQA Preparation and Manifests

Status: implemented and verified, including the full local corpus; results are below.
Duration: ten working days. Team: project owner + Codex. Date: 2026-09-23.
Dataset source: `/Users/symperaai/Code/StegoLab/data`, selected by the owner.
Local UHD-IQA files are present in `data/UHD-IQA-database`; no download work is included.
The reviewed inventory contains all 6,073 source images and their metadata.

## Current project status

- Sprint 1 and Sprint 2 are implemented and merged into local `main`.
  The reviewed checkout is `abcef42`, which also includes the full architecture guide.
- Working: four tabs, Config save/reset, persistent settings and job metadata,
  strict contracts, CPU containers, encrypted message framing, error correction,
  test payload maps, image preparation, and public command-line demonstrations.
- Sprint 3 adds local dataset preparation and offline manifests. Still missing:
  dataset downloads, worker execution, live events, models,
  training, independent model exports, and image-based encode/decode.
- Protocol tests prove exact byte recovery through a test channel. Image tests
  prove prepared PNG pixel preservation. Neither proves neural image-message recovery.
- Latest review checks at `abcef42`: 192 backend tests and 10 frontend tests pass; exported
  contracts match; all 82 checked source files contain fewer than 300 lines.
- Both existing containers are healthy. The API reports CPU only, no installed
  models, no available model operations, and zero usable payload capacity.
- The backend tests emit the existing non-blocking Starlette/httpx warning.
  Browser restart, clean container builds, lint/type checks, and GPU checks were
  not repeated in this planning review; earlier acceptance evidence remains in
  [Sprint 1](sprint_01.md) and [Sprint 2](sprint_02.md).

## Goal and scope

- Take a folder of local images, validate it, save prepared copies, and produce
  a frozen manifest: a record of the exact images, checksums, rules, and splits.
- Give later training code a repeatable input that works offline and can detect
  changed or missing files. Finish with a small CPU demonstration.
- This is the first slice of Step 3 in the [execution order](execution_order.md).
  It enables local model development; it does not complete all dataset adapters
  or freeze the full research pilot dataset.
- Deliver backend services, strict schemas, command-line tools, a container source
  mount, focused tests, and a dataset guide. Keep the current GUI behavior.
- Defer browser uploads, dataset HTTP routes, Hugging Face/HTTPS application
  downloaders, archive extraction, shared download cache, worker scheduling,
  live progress events, pause/resume, models, training, and cloud work.
- Text corpora are optional future fixtures. The planned binary-payload model
  trains with random bits; image/text pairing is not required for this sprint.
- No GPU hours, model-quality targets, or SOTA claims belong to this sprint.

## Selected images and input readiness

- Inventory on 2026-09-23: **6,073 JPEGs, 10,713,580,611 source image bytes**.
  Official split counts are 4,269 training, 904 validation, and 900 test. There
  are 5,953 RGB and 120 grayscale images; all names match metadata. Width is
  3,840; heights range from 828 to 3,826. No source header exceeds dataset limits.
  5,114 images exceed the unchanged production area limit. The metadata SHA-256
  is `90a46279b4532689efb7959f44c53ffdd902d43870d6532de87291909ccd2940`.
- The [official UHD-IQA page](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html)
  describes 6,073 photos, resized to width 3840 with aspect ratio preserved,
  and CC0 image terms. Its metadata includes `image_name`, `set`, `subset`, and
  `quality_mos`. Keep the dataset citation and terms reference in provenance.
  These are published facts, not a count of the local folder.
- Use extracted images and `uhd-iqa-metadata.csv` under `data/`; inspect the
  actual folder layout before choosing image and metadata paths. Treat `set`
  as the source split; preserve `subset`. Map the observed train/validation/test
  labels to train/tuning/held-out explicitly and record that mapping.
- Add a small local UHD-IQA metadata reader. Join by exact dataset image name,
  namespace identities as `uhd_iqa:<image_name>`, and hash the metadata bytes.
  Duplicate or ambiguous keys, unknown split labels, unsafe names, and selected
  images without metadata are errors. Require metadata for UHD-IQA mode; never
  silently generate new splits. Optional MOS is source image-quality metadata,
  not a hidden-message label or a measure of steganographic quality.
- Inventory headers before pixel decoding: report file count, bytes, dimensions,
  modes, and limits. Reconcile local names with metadata; report missing and
  extra files. A declared local subset is valid, but must not be called the full
  dataset. A full-dataset claim requires all expected identities to be present.
- Use the source read-only. Native prepared output goes to repository
  `datasets/uhd_iqa/<revision>/`; container output goes to
  `/data/datasets/uhd_iqa/<revision>/`. Neither is inside the source folder.
  Exclude `/data/` from Git and the Docker build context; mount it at runtime.
- The real-data sample is frozen in `docs/examples/uhd_iqa_sample.json`;
  synthetic fixtures cover malformed files and failure paths without modifying originals.

## User stories

- As an ML engineer, I can prepare a local image folder with one command and
  see accepted, excluded, duplicate, and rejected counts with clear reasons.
- As a researcher, I can reuse the same data revision and split membership,
  so comparisons do not silently change their inputs.
- As an operator, I can interrupt a run or recover from a failed write without
  losing source images or an already completed dataset.

## Backlog and delivery order

| Days | Work | Required evidence |
|---|---|---|
| 1–2 | Inventory `data/`; freeze schemas, UHD-IQA metadata mapping, image limits, split rules, and failures. Build small synthetic fixtures. | Actual counts or missing-input report; reviewed records; separate source and production image policies; large-image memory probe. |
| 3–4 | Implement bounded folder scanning and image preparation. | Correct format/color/orientation/alpha handling; invalid files are reported; source files remain unchanged. |
| 3–5, parallel | Implement UHD-IQA metadata reader, identities, exact duplicate groups, split assignment, and manifest validation. | Same input/settings produce the same revision; official splits are preserved; duplicate conflicts fail. |
| 5–6 | Add persistent staging, dataset publication, integrity checks, and safe reruns. | Partial output is never reported complete; interrupted and failed runs preserve existing revisions. |
| 7–8 | Integrate CLI commands, read-only source mounting, disk checks, and documentation. | The same prepared dataset works after the original source is unmounted and with networking disabled. |
| 9–10 | Run acceptance, independent review, fixes, and the UHD-IQA demonstration. | All sprint gates pass; record local coverage, measured time, storage, memory, and remaining limits. |

- Codex owns contracts and integration. Separate agents can own local image
  preparation, manifests/splits, and independent review after Day 2 contracts.
- Use feature branches such as `codex/sprint-03-dataset-preparation`,
  `codex/sprint-03-dataset-manifests`, and `codex/sprint-03-dataset-integration`.
- The owner reviews the contracts early and the demonstration at sprint end.
  Contract changes update the schema/policy version and fixtures together.

## Dataset image rules

- Do not pass every source through the current production `ImageDimensions` or
  `ImageSummary`: both require at least 512 pixels per side. Training is planned
  around 256-pixel crops, and useful dataset sources can be smaller than 512.
- Add separate strict dataset image records and a trusted internal validation
  policy. Reuse the existing format, color, orientation, and bounded-decoding
  logic without weakening production image commands or public capacity rules.
- Fixed dataset source bounds: sides of 1–8192 pixels, area at most
  32,000,000 pixels, and file size at most 50 MiB. Prepared PNGs may be up to
  128 MiB; their reader uses this separate allowance. Keep one image in processing
  at a time and close unused full-pixel buffers before output verification.
- For example, 3840 × 2560 is 9,830,400 pixels and exceeds the current production
  area limit. This illustrates the policy gap, not a measured local image.
  Production limits remain 512–4096 per side and 8,850,000 pixels. Report sources
  above the dataset limits as unsupported; do not silently crop or resize them.
- Accept supported single-frame, eight-bit L/RGB JPEG and L/RGB/RGBA PNG.
  Expand grayscale as R = G = B and record the conversion. Grayscale embedded
  ICC profiles are unsupported; all 120 local grayscale images are untagged. Apply
  orientation once, use the existing sRGB policy, preserve alpha, strip source
  metadata from prepared output, and verify saved/reopened pixel equality.
- Preserve prepared dimensions during import. Record sources smaller than
  256 pixels on either prepared side as ineligible for the initial training
  crop policy. Do not silently enlarge them to increase accepted counts.
- Invalid/unsupported files receive a reason in the report. Report valid but
  ineligible files separately. An empty or all-ineligible input fails clearly.
- Create training crops and augmentations later, after split membership is
  frozen. Declare benchmark resizing/cropping separately; imported sources
  are not automatically native 1024-pixel benchmark images.

## Contracts and reproducibility

| Record | Required contents |
|---|---|
| `DatasetPreparationRequest` | Schema version, dataset name, local source, preparation policy, split policy/seed, source identity/split map (required in UHD-IQA mode), metadata reference, and terms reference. |
| `DatasetImageRecord` | Relative source and prepared paths, namespaced upstream identity when known, upstream split/subset and assigned split, duplicate/split groups, exact-duplicate representative reference, source/prepared/RGB-pixel checksums, dimensions, preparation decisions, and eligibility. |
| `DatasetManifest` | Schema and policy versions, dataset revision, software provenance, source URL and metadata checksum, split settings, local coverage, counts/storage, records/report checksums, and review/readiness status. |
| `DatasetSummary` | Safe completion/integrity result and counts by split, eligibility, duplication, and rejection reason. |

- Store a small manifest plus validated per-image records and a rejection report.
  Stream image processing and large record files; never load all image pixels
  into memory. Freeze selected source identities and split mapping with the
  revision, so source metadata is not needed for offline reuse. Keep images and
  prepared data outside Git.
- Use SHA-256 for source bytes, prepared PNG bytes, and model-visible RGB pixels.
  Include dimensions in the pixel digest; alpha differences must not place
  identical RGB covers in different splits.
- Form exact duplicate groups before assigning splits. Choose one deterministic
  representative for training and retain alias/source provenance. Also group
  records with the same namespaced upstream image identity, even when their pixels differ.
  All members of either relationship must share a split. Keep nonidentical
  variants as separate examples; conflicting declared splits are a blocking
  error. Build split groups across both relationships, including linked chains.
- Only for explicitly unlabelled local development data, use a versioned, seeded hash of each stable group
  identity with 80% train, 10% tuning, and 10% held-out ranges. Record actual
  counts. Publication requires at least one eligible, unique training image.
  Empty tuning or held-out splits are allowed with explicit warnings; they do
  not support evaluation readiness. Add images before a full train/tune/test run.
- UHD-IQA source split maps take precedence over generated development splits.
  The current unlabelled-local request supports generated splits only; arbitrary
  explicit local split maps remain future work.
  Keep upstream and assigned split names separate, and never overwrite an
  existing revision or reshuffle it when new files are added. UHD-IQA uses its
  metadata splits, not the generated 80/10/10 rule. Never move held-out images
  into training to fill an empty split or resolve a duplicate conflict.
- Define canonical record ordering and serialization. The same relative source
  tree, bytes, metadata, policies, seed, and supported tool versions produce the same
  revision regardless of traversal order or absolute source-root location.
  Exclude run timestamps and absolute paths from the revision identity.
- Hash and decode the same bounded source bytes. Detect source replacement or
  modification during scanning; fail or record a changed-source rejection.
  Completed data must depend only on its prepared copies and frozen records.
- Rehash prepared files and validate records before reuse. Recompute revision
  identity and check counts, eligibility, representative references, and split
  groups across records. Tampering, missing files, path escapes, unsupported
  versions, and inconsistent group/split/checksum declarations fail.
- Define a shared reader that selects eligible exact-duplicate representatives
  for one requested split, using only paths inside the completed revision.
  An empty split returns no examples and a clear warning; it never borrows
  examples from another split. This is the later training-loader contract.
- Exact duplicate checks do not detect every resized or recompressed copy.
  Keep `pilot_ready` false in this sprint. Full pilot readiness requires a
  separate review of canonical identities, near duplicates across all sources,
  source terms, and frozen benchmark membership before pilot training starts.
- Preserve the planned benchmark: canonical COCO validation plus 5,000 untouched
  training-source images, with separate native high-resolution results. A local
  UHD-IQA demo does not replace those identities or satisfy that gate. Reserve
  UHD-IQA held-out images for separate later high-resolution evaluation.

## Storage, limits, and interruption

- Store prepared revisions under `datasets/<dataset_name>/<revision>/` in the
  persistent data area. Use file manifests as the authority in this slice;
  no SQLite migration, job execution, or durable event stream is required.
- Use `/data/datasets` in the container and a documented local dataset root
  for native commands. Keep the existing settings directory `/data/state`
  and its `STEGOLAB_DATA_DIRECTORY` setting unchanged.
- Add an opt-in Compose override mapping the selected host `data/` read-only
  to `/sources/uhd-iqa` inside the backend. Explain host versus container paths
  and permissions for the existing non-root user. Preserve the two-container setup.
- Keep staging on the same persistent filesystem as the final dataset, never
  in the backend's 64 MiB `/tmp`. Verify every required output before publishing
  the final revision atomically without replacing an existing revision.
- Check name/path containment; reject symlinks and non-regular files. Refuse
  source/output overlap. Import no scripts, archives, or remote URLs.
- Initial run ceilings: 200,000 scanned entries, 100 GiB source bytes, and
  100 GiB prepared bytes. Count rejected candidates toward scanning/read limits.
  Keep one image in processing at a time; enforce limits during work as well
  as preflight. Record the effective limits in the preparation report.
- Measure PNG expansion on the local sample before scheduling the full import.
  The full available corpus must fit both the output ceiling and disk reserve;
  otherwise report the required space. A limit reached during import fails that
  run without publishing partial data. A smaller run needs an explicitly selected,
  named subset frozen before import; never label that revision full preparation.
- Preserve at least 10 GiB or 10% of the destination filesystem, whichever is
  larger. Estimate decoded PNG expansion and temporary writes before starting;
  check the next bounded write against that reserve throughout staging and
  publication. Serialize imports per destination root in this first version,
  so concurrent imports cannot each spend the same free-space allowance.
  Report insufficient space before completion.
- Use one destination-root writer lock, held from preflight through publication
  and cleanup. Identical reruns compare source hashes and validate/reuse an
  existing revision before preparing new images. Changed inputs or settings create
  new revisions. Concurrent writers cannot clean up one another's staging.
- On Ctrl-C/termination, stop at a safe boundary and clean only this run's
  staging. A crash may leave owned staging; the next invocation detects it and
  cleans it safely after checking ownership/locks. No automatic resume or
  fixed three-second cleanup promise is part of this slice.
- Disk-full, permission, allocation, and cleanup failures return clear errors.
  Preserve the source and previous completed revisions. Logs contain safe run
  metadata, not image metadata, file contents, or absolute source paths.

## Interfaces and demonstration

- Implement `prepare_dataset`, `inspect_dataset`, and `validate_dataset` through
  shared services: `prepare_dataset <request.json>`, `inspect_dataset <revision>`,
  and `validate_dataset <revision>`. Preparation also accepts `-` for stdin.
  Return codes are 0 success, 1 processing/integrity failure, 2 invalid input,
  130 Ctrl-C, and 143 termination.
- `prepare_dataset` accepts a local source and validated preparation settings;
  `inspect_dataset` reports manifest information; `validate_dataset` performs
  the full file/checksum integrity check. Clearly distinguish inspection from
  verified integrity. Return structured summaries and useful exit codes.
- Export new entity schemas and keep the existing HTTP contracts stable.
  Leave `train`, `encode`, and `decode` unavailable and public capacity at zero.
- Demo: prepare mixed valid/invalid images → inspect counts/reasons → verify
  immutable splits → rerun/reuse → unmount the source → validate offline →
  show safe failure after tampering with a disposable prepared copy.
- Use a frozen local UHD-IQA sample of up to 30 images (up to 10 per available
  upstream split, at least one eligible training image). Choose deterministically
  from metadata; include the largest supported area and varied aspect ratios
  within each split, then fill by image name. Save identities and selection rules.
  The frozen 30-image sample also includes grayscale and largest-file examples
  in every split. Run failure cases on synthetic fixtures or disposable copies,
  never originals.
- Inventory the full available local corpus and run its bounded preparation when
  space permits. Record expected/present/accepted/rejected counts by source split,
  the revision, elapsed time, peak memory, and source/prepared bytes. Sample-only
  acceptance must state its coverage and the reason full preparation is pending.
- Add `docs/datasets.md` with native/container commands, path meanings,
  eligibility rules, manifest examples, disk planning, cleanup, and limitations.
  Update README and architecture status during implementation; run the existing
  Archify validation workflow if its artifact changes.

## Definition of done

| Gate | Required result |
|---|---|
| UHD-IQA input and splits | Local images and metadata exist; sample identities and local coverage are recorded; source splits/subsets survive import; missing or conflicting metadata fails clearly. |
| Image behavior | Orientation/color/alpha and small dataset sources work under the dataset policy; existing production image limits and protocol tests still pass. |
| Reports | Valid, invalid, ineligible, and duplicate counts are correct; empty/all-ineligible inputs fail without false completion. |
| Reproducibility | Reordered discovery and relocated source roots produce the same revision and splits; changed input creates a new revision. |
| Leakage controls | Renamed files, metadata variants, and identical RGB/different-alpha files stay in one duplicate group; variants with the same upstream identity share a split; conflicting declared splits fail. |
| Integrity | Missing/tampered prepared files, invalid revision identities, and inconsistent split/group/representative records fail validation; completed revisions work without source files. |
| Training handoff | The shared reader selects only eligible representatives in the requested split; paths remain inside the revision and empty splits never fall back to other data. |
| Failure recovery | Disk-full, unreadable files, unsafe paths, concurrent writers, Ctrl-C, and abandoned staging preserve source and completed revisions. |
| Resource bounds | File/byte/pixel limits and free-space reserve hold throughout staging/publication, including concurrent imports; large-image probe and named UHD-IQA sample have measured time, peak memory, and output size. |
| Container demonstration | Non-root import from a read-only source succeeds; completed data survives restart and validates with networking disabled. |
| Regression and delivery | Focused dataset tests, existing protocol/image tests, schema checks, lint/types, file-length checks, locked CPU build, and existing browser restart workflow pass. |
| Honest readiness | No trained-model or benchmark claim; remote adapters, near-duplicate pilot audit, and final pilot manifests remain explicitly open. |

- Keep routine CI small: synthetic/public fixtures and one CLI integration flow;
  no live dataset download, large corpus, GPU workload, or 10,000-image evaluation.
- Record acceptance below with the source baseline, environment, exact commands,
  results, and known limitations. Only recorded completed checks are passed.
- Follow-on work: remote adapters and their security/cancel/cache gates can run
  alongside model development once this local dataset contract is stable. The
  GPU pilot still waits for complete frozen data, model/export, and infrastructure gates.

## References

- Owner's local context: `plan/CONTEXT.md`; [backend plan](backend.md),
  [frontend plan](frontend.md), [infrastructure plan](infrastructure.md),
  [execution order](execution_order.md), and [architecture](../Architecture.md).
- Existing [image guide](../docs/image_preparation.md) and
  [protocol guide](../docs/message_protocol.md) remain the production baseline.


## Implementation acceptance — 2026-09-23–24

Source baseline: `abcef42`; implementation is on
`codex/sprint-03-dataset-preparation`. Existing user planning changes were retained.
Native environment: macOS 26.6.2 arm64, Python 3.12.13, locked dependencies.
Container environment: Linux arm64, Python 3.12.14, locked CPU images.
No GPU time or remote dataset download was used.

| Check | Recorded result |
|---|---|
| Backend tests | 291 pass, including protocol/image regressions and dataset failure cases |
| Frontend | 10 tests pass; TypeScript, ESLint, Prettier, and production build pass |
| Python quality | Ruff, formatting, strict mypy, schema drift, and git whitespace checks pass |
| File length | 111 checked source files; every file is below 300 lines |
| Browser | Save/restart/reload/reset and accessibility flow passes on disposable port 8083 |
| CPU containers | Both locked images build and both services become healthy |
| Native sample | 30 accepted, no rejections; 10 per split; 355,293,227 prepared bytes; 19.65 seconds; 341.98 MiB peak process memory |
| Container sample | 30 accepted, no rejections; 362,842,613 prepared bytes; 64.70 seconds; 288.36 MiB peak process memory |
| Offline container | Sample validates with `--network none` and no source mount |
| Container reuse | Same sample revision reused after container recreation, without new prepared images |
| Maximum-size probe | 8000×4000 RGB JPEG: 37,529,978 source bytes → 95,949,449 PNG bytes; exact verification; 807.59 MiB peak including fixture creation |
| Architecture | Archify 9/9 showcase checks, four desktop sizes without overflow, light/dark and SVG visual review pass |
| Full local corpus | 6,073 accepted and verified; no rejections, ineligible images, or exact RGB duplicates; 60,898,329,594 prepared bytes; 3,879.61 seconds; 329.56 MiB peak process memory |

Native sample revision:
`c3b9eb1bea77b4172a620814f047e5214d89b24ef53639582980546cfb20d0d2`.
Container sample revision:
`e932d96aa54e120351df272474bf614c0572544ef493425392a5deca652e2c03`.
Full local revision:
`75bef6dab2b3de136d1c5c359d6753a25e4a750d5387fe75921e394482f0776c`.
It was atomically published under `datasets/uhd_iqa/<revision>/` after the complete
PNG checksum and pixel scan passed, at 2026-09-23 21:11:24 UTC
(2026-09-24 00:11:24 Asia/Jerusalem). The full run took 64 minutes 39.61 seconds.

- Coverage is complete: 4,269 train, 904 tuning, and 900 held-out images;
  all 6,073 remain eligible representatives. No source images were excluded.
- All 120 grayscale images were converted and recorded: 84 train, 23 tuning,
  and 13 held-out. All source dimensions are preserved.
- Source image bytes: 10,713,580,611. Prepared PNG bytes: 60,898,329,594
  (56.72 GiB). Preflight estimate: 84,799,899,050 bytes; actual output stayed
  within the 100 GiB ceiling and the free-space reserve.
- The final code also passed canonical metadata, grouping, coverage, and file
  inventory validation on the published full revision. The staging folder is empty.
- Detailed native measurements are in
  `logs/2026-09-23_21-11-24_fe1906d2748049abbce0ee4523f98488/dataset_run.json`.
  Run reports and prepared images remain local and are excluded from Git.

Encoder/runtime differences are recorded in provenance and can change prepared
bytes and revision identity. The 32-million-pixel probe is outside routine CI;
its measured time includes disposable seeded fixture creation.

Commands used for repeatable checks:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest tests/backend
uv run --locked python -m backend_service.export_contracts --check
uv run --locked python scripts/check_file_lengths.py
uv run --locked stegolab verify_protocol
uv run --locked stegolab prepare_dataset docs/examples/uhd_iqa_sample.json
uv run --locked stegolab prepare_dataset docs/examples/uhd_iqa_full.json
uv run --locked python scripts/measure_dataset_resources.py
npm --prefix user_interface run lint
npm --prefix user_interface test
npm --prefix user_interface run build
COMPOSE_PROJECT_NAME=stegolab-sprint03-check STEGOLAB_PORT=8083 STEGOLAB_BASE_URL=http://127.0.0.1:8083 STEGOLAB_RESTART_BACKEND=1 npm --prefix user_interface run test:browser
```

Container import and offline commands are documented in [datasets](../docs/datasets.md).
Acceptance used Compose project `stegolab-sprint03-check`, its own volume, and
port 8083. The disposable project and volume were removed after checks. The existing
application on port 8081 was not restarted or reset. These acceptance measurements
ran locally; pull-request CI results are reported separately by GitHub Actions.
The existing Starlette/httpx deprecation warning remains non-blocking.
Near-duplicate review, the COCO benchmark, remote adapters, models, training,
and GUI uploads remain open; `pilot_ready` is false.
