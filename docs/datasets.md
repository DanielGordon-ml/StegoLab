# Prepare local image datasets

Sprint 3 provides local, offline dataset preparation. It does not train a model
or hide text inside an image. Browser uploads and remote downloads remain future
work. Commands share strict backend services and never change the source images.

## Local UHD-IQA inputs

The selected source is `data/UHD-IQA-database/`. It contains `training/`,
`validation/`, `test/`, `uhd-iqa-metadata.csv`, and the optional tags file.
The local inventory has 6,073 JPEGs: 4,269 training, 904 validation, 900 test.
Of these, 120 are grayscale. All filenames match the supplied metadata.

Keep the [official citation and image terms](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html)
with any research using this dataset. The importer stores this reference and the
metadata checksum. It performs no downloads or archive extraction.

## Commands

From the repository root, with locked Python dependencies installed:

```sh
uv run --locked stegolab prepare_dataset docs/examples/uhd_iqa_sample.json
uv run --locked stegolab prepare_dataset docs/examples/uhd_iqa_full.json
uv run --locked stegolab inspect_dataset datasets/uhd_iqa/REVISION
uv run --locked stegolab validate_dataset datasets/uhd_iqa/REVISION
```

Replace `REVISION` with the 64-character revision returned by preparation.
Inspection reads the manifest and returns `integrity: "not_checked"`.
Validation checks records, splits, files, hashes, and decoded PNG pixels and
returns `integrity: "verified"`. It does not need the original source folder.
Identical inputs/settings reuse a matching revision only after full validation;
changed inputs create a new revision. Existing revisions are never overwritten.

Preparation takes a JSON filename, or `-` to read JSON from standard input.
The example files contain the frozen 30-image sample and the full local request.
Paths in requests are relative to the command's working directory. The metadata
path and selection paths are relative to `source_directory`.

```json
{
  "schema_version": 1,
  "dataset_name": "uhd_iqa",
  "source_kind": "uhd_iqa",
  "source_directory": "data/UHD-IQA-database",
  "metadata_file": "uhd-iqa-metadata.csv",
  "output_root": "datasets",
  "policy_version": "dataset_v1",
  "seed": 0,
  "selection_name": null,
  "selection": null
}
```

Unknown fields and implicit type conversion are rejected. A reduced run requires
both a safe `selection_name` and a nonempty list of unique relative image paths.
The sample chooses ten images per official split: largest area, widest/tallest
aspect ratios, largest source file, first grayscale name, then remaining image
names in lexical order. Ties use image name; repeated choices count only once.
Its selected names are frozen in the example JSON, not chosen again on each run.

For unlabelled development images, set `source_kind: "local"`, choose a dataset
name, and set `metadata_file: null`. This mode currently supports generated splits
only. Omitted source/terms references become `local` and `not_reviewed`; supply
your own references when available. A seeded hash of each connected identity and
pixel group assigns 80/10/10 ranges to train/tuning/held_out. Small datasets may
have empty evaluation splits, which produce explicit warnings. At least one
eligible unique training example is required. These development splits never
replace UHD-IQA's metadata splits.

| Exit code | Meaning |
|---|---|
| 0 | Successful preparation, reuse, inspection, or validation |
| 1 | Source, storage, split, or integrity failure |
| 2 | Invalid command arguments or request JSON |
| 130 | Ctrl-C; owned staging is cleaned through normal shutdown |
| 143 | Termination signal; owned staging is cleaned through normal shutdown |

Failures do not print parser details, source metadata, or absolute source paths.
Each command writes safe event logs. Successful preparation also writes a
`dataset_run.json` with elapsed time, peak process memory, counts, and limits under
`logs/<date_and_run>/`, or `STEGOLAB_LOG_DIRECTORY` when configured.

## Image rules

| Rule | Dataset preparation | Existing production image commands |
|---|---|---|
| Sides | 1–8,192 pixels | 512–4,096 pixels |
| Area | At most 32,000,000 pixels | At most 8,850,000 pixels |
| Source bytes | At most 50 MiB | At most 50 MiB |
| Prepared PNG bytes | At most 128 MiB | Existing production policy |
| Sources | Eight-bit L/RGB JPEG; L/RGB/RGBA PNG | RGB JPEG; RGB/RGBA PNG |

- Grayscale conversion copies each gray value to all three RGB channels.
- Orientation is applied once. Supported color declarations use the existing
  sRGB conversion rules. Grayscale with an embedded color profile is unsupported;
  all 120 local grayscale images have no such profile.
- Prepared dimensions stay unchanged after orientation. Supported alpha stays
  unchanged; source metadata is removed. Saved/reopened pixels must match exactly.
- Sources smaller than 256 on either prepared side remain recorded but are
  ineligible for the initial crop-based training policy. No silent enlargement,
  resizing, or cropping occurs.
- Invalid/unsupported candidates are reported. CSV sidecars, tags, `.DS_Store`,
  and `__MACOSX` files are not image examples. Unsafe links and special files fail.

## Splits, duplicates, and revisions

UHD-IQA `training`, `validation`, and `test` map to `train`, `tuning`, and
`held_out`. The original split, subset, and namespaced image identity stay in
each record. Missing or ambiguous metadata and conflicting declared splits fail.

Exact duplicate detection uses prepared RGB pixels and dimensions, ignoring
alpha. Identical RGB images share one deterministic representative. Sources
sharing an upstream identity also share a split, including chains connecting
both relationships. Nonidentical variants remain separate examples.

Each completed revision contains:

```text
datasets/<dataset_name>/<revision>/
  manifest.json
  records.jsonl
  rejections.jsonl
  images/<source_checksum>.png
```

Records contain relative paths, source/prepared/pixel SHA-256 values, dimensions,
mode changes, eligibility, groups, and split membership. The manifest records
software and policy versions, source coverage, counts, and record-file hashes.
Canonical UTF-8 JSON uses sorted keys and one final newline. Records are sorted
by relative source path. Revision identity excludes absolute paths, timestamps,
run identifiers, and resource measurements. Different supported encoder versions
can produce different bytes and revisions; provenance records those versions.

The shared `backend_service.dataset_reader.read_dataset(directory, split)` returns
only eligible exact-duplicate representatives from that split, after validation.
An empty split returns no examples and a warning. It never substitutes another
split. Training crops and augmentation belong to later work.

Exact matching does not detect every resized or recompressed duplicate. The
perceptual-hash audit below reports those; `pilot_ready` still remains false, and
a source-terms review stays separate. UHD-IQA does not replace the COCO benchmark
in the backend plan.

## Fetch remote sources from the command line

Sprint 7 adds two commands that download a source, save its raw files under
`data/<source_name>/` (next to any folder you place there yourself), and then
prepare a revision under `datasets/` exactly like a local folder:

```sh
uv run --locked stegolab inspect_source docs/examples/inspect_imagenet.json
uv run --locked stegolab fetch_dataset docs/examples/fetch_div2k_validation.json
uv run --locked stegolab fetch_dataset docs/examples/fetch_coco_validation.json
uv run --locked stegolab fetch_dataset docs/examples/fetch_wikitext.json
```

- Sources are Hugging Face repositories (pinned to a commit), plain https
  archives, or archives uploaded through the browser. Only https is accepted;
  private, local and metadata addresses are refused, on every redirect too.
- Inspection resolves the revision and reports sizes, access and pause support
  without downloading. A gated repository needs `HF_ACCESS_TOKEN` (or
  `HF_TOKEN`) in the backend environment; the command then explains this and
  downloads nothing.
- Downloaded archives are kept, checksum-verified, under
  `.cache/stegolab/datasets/`. A second run of the same request downloads
  nothing and reuses the raw folder and the prepared revision.
- Archive members are extracted only when their paths are safe, their bytes
  match their declared type and size, and the limits hold; everything else is
  recorded as a rejection. `archive_splits` maps member folders to splits, and
  `training_intended: false` allows evaluation-only revisions such as COCO
  validation. Text sources such as wikitext are saved as `.txt` files and are
  not prepared as images.
- Ctrl-C or a termination signal keeps the partial download for a later resume
  and removes only the run's staging folders. The near-duplicate audit below is
  a separate step; fetching never makes a dataset pilot-ready.

## Benchmark identities and near-duplicate audit

Sprint 7 adds three research commands. They write evidence files only: nothing
is trained, evaluated or approved, and `pilot_ready` stays `false` everywhere.

```sh
uv run --locked stegolab fetch_dataset docs/examples/fetch_coco_annotations.json
uv run --locked stegolab freeze_benchmark docs/examples/freeze_benchmark.json
uv run --locked stegolab validate_benchmark docs/benchmarks/release_benchmark_v1 --annotations-directory data/coco2017_annotations/annotations
uv run --locked stegolab audit_near_duplicates docs/examples/audit_uhd_iqa_smoke.json
uv run --locked stegolab audit_near_duplicates docs/examples/audit_uhd_iqa.json
```

**What is frozen and why.** The release benchmark is a fixed list of COCO 2017
image identities: all 5,000 `val2017` images as `validation`, plus 5,000
`training_source` and 1,000 `reserve` images chosen from `train2017` in a
seeded, deterministic order. Freezing the list means later benchmark runs, and
any later preparation policy, cannot quietly change which images are measured.
Only identities are frozen (member path, COCO identifier, declared size); no
pixels are copied and `preparation_policy` stays `"not_frozen"`.

`freeze_benchmark` reads `captions_val2017.json`, `instances_val2017.json` and
`captions_train2017.json` from the fetched annotations folder and writes
`identities.json` and `members.jsonl` under `docs/benchmarks/release_benchmark_v1/`.
It never overwrites an existing folder, and identical inputs with the same seed
give byte-identical files. `validate_benchmark` re-checks both checksums, the
counts and the member order without changing the folder; with
`--annotations-directory` it re-derives the list and reports
`verified_with_annotations`. The example request pins the three mirror archives
by commit, path and size; its `archive_sha256` values are sixty-four zeros as a
placeholder, so replace each with the fetch summary or Hub tree checksum first.

Fetching the annotations zip keeps its `.json` members as text files under
`data/coco2017_annotations/annotations/`. A `.json` member must start with `{`
or `[` after an optional byte-order mark and whitespace, and the 256 MiB text
limit applies: `instances_train2017.json` is recorded as a `member_size_limit`
rejection, while the three files the freeze needs are smaller and pass.

**Near-duplicate audit.** `audit_near_duplicates` hashes every exact-duplicate
representative of one prepared revision with a 64-bit perceptual hash
(`phash_dct_32_v1`: grayscale, 32×32 box resize, DCT, top-left 8×8 block against
its median). It never writes inside the revision, and each image is checked
against its recorded checksum when read. The report counts pairs at four
Hamming distance levels, 0, 4, 8 and 12, within each split and across splits
(`train_tuning`, `train_held_out`, `tuning_held_out`), and lists the pairs at
distance 12 or less, sorted and capped at 2,000. Identical images sit at 0,
resized or recompressed copies usually at 4 or less, and unrelated images well
above 12. A shifted crop is not detected and needs another review.

Reports live outside the revision at
`<output_root>/.audits/<dataset_name>/<audit identifier>/near_duplicate_audit.json`.
`output_root` defaults to the folder holding the dataset name (`datasets/` for
`datasets/uhd_iqa/<revision>`), and the identifier is the first twelve
characters of the revision plus a UTC start time, so every run gets a new folder.
An `output_root` that is a revision or a dataset name folder is refused before
any work, because a stray folder there would spoil the immutable revision or
the revision listing, and the report folders are created without following
symbolic links. The report carries its own checksum, the revision and records
checksum it describes, elapsed time, peak memory and library versions. With
`benchmark_identities_directory` it also records the frozen identities
checksum; because no benchmark pixels are prepared yet, that block reports
`pixels_unavailable`. A `benchmark_dataset_directory` needs that identities
folder too: its records are matched to members by the member path in their
identity (the source path for a locally prepared folder), and a revision that
matches no member is refused rather than reported as compared. A `limit` (200
in the smoke example) hashes only the first N representatives and marks the
report `limited`. Pair counts have no practical limit; only the listed pairs
are capped at 2,000, and the report says when that cap was reached.

**The automated gate.** Preflight fails when any pair crosses splits at
distance 8 or less, when the report describes another revision, or when it is
limited or altered. Distance 8 is the automated rule; the 0, 4 and 12 counts
support a human review. A passing gate is evidence about one revision, not a
readiness decision: `pilot_ready` stays `false`, and the benchmark pixels, the
1024 preparation policy and the evaluation remain later gates.

## Storage and recovery

- One process imports into a given output root at a time. A second writer fails
  clearly rather than sharing the available disk budget.
- Per-run limits are 200,000 scanned entries, 100 GiB source bytes, and 100 GiB
  prepared writes. Rejected files and sidecars count toward scan/source limits.
- Preflight estimates PNG expansion, while every actual write checks the byte
  ceiling and preserves at least 10 GiB or 10% of the filesystem, whichever is
  larger. The expansion estimate is not a promised final file size.
- Measure the sample before a full import. PNGs can be much larger than JPEGs.
  A run exceeding a limit fails without publishing a partial dataset. Select a
  named subset when the full corpus does not fit; never label it full coverage.
- Staging lives under `output_root/.staging` on the publication filesystem.
  It never uses the container's small `/tmp` for image storage.
- Ctrl-C and SIGTERM clean only this run's staging. A hard crash may leave a
  marked stage; the next import recovers owned staging while holding the root
  lock. Unknown/unowned files are preserved. There is no automatic resume.
- Source and output trees must be separate, with no symlink parents. Completed
  data is published atomically without replacing an existing revision. Do not
  edit completed files; integrity validation detects changed or missing content.

## Containers and offline verification

The optional override mounts the local dataset read-only at `/sources/uhd-iqa`:

```sh
docker compose -f infrastructure/compose.yaml -f infrastructure/compose.datasets.yaml build backend_service
python3 - <<'PY' | docker compose -f infrastructure/compose.yaml -f infrastructure/compose.datasets.yaml run --rm -T --no-deps backend_service python -m backend_service.command_line prepare_dataset -
import json
from pathlib import Path
request = json.loads(Path("docs/examples/uhd_iqa_sample.json").read_text())
request.update(source_directory="/sources/uhd-iqa", output_root="/data/datasets")
print(json.dumps(request))
PY
```

Set `STEGOLAB_SOURCE_DIRECTORY` to an absolute host dataset path if needed.
The source must be readable by container user 10001. Prepared data goes to the
persistent volume under `/data/datasets`; configuration remains `/data/state`.
Native `datasets/` and the Docker volume are different destinations.

After preparation, verify without mounting any source or enabling networking:

```sh
docker run --rm --network none --read-only --memory 10g \
  --tmpfs /tmp:size=64m --env STEGOLAB_LOG_DIRECTORY=/data/logs \
  --volume stegolab_application_data:/data stegolab-backend_service \
  python -m backend_service.command_line validate_dataset /data/datasets/uhd_iqa/REVISION
```

Adjust the image/volume names if using another Compose project. Acceptance and
browser restart tests use a disposable project so working settings are preserved.
Do not remove the working application's volume when stopping test containers.
