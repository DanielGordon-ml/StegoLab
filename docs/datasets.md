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

Exact matching does not detect every resized or recompressed duplicate. Therefore
`pilot_ready` remains false. A separate near-duplicate, source-terms, and benchmark
review is required before the research pilot. UHD-IQA does not replace the COCO
benchmark in the backend plan.

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
