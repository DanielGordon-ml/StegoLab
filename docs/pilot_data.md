# Pilot dataset and reproducible crops

- The loader references the existing full UHD-IQA revision
  `75bef6dab2b3de136d1c5c359d6753a25e4a750d5387fe75921e394482f0776c`.
  It never copies or rewrites the revision.
- Canonical manifests, record checksums, group membership, split counts and file
  inventory are checked first. This step reads metadata, without decoding images.
  Prepared PNG checksums are checked when a training crop is requested.
- All 4,269 unique training covers participate in shuffled epochs. The synchronous
  sampler reads each requested image and makes a random 256×256 RGB crop. There
  are no background workers or prefetched samples.
- Tuning keeps its 904 identities. Of these, 902 support native 1,024×1,024 crops.
  `uhd_iqa:3125.jpg` and `uhd_iqa:774.jpg` are excluded because both are 3840×960.
  Held-out images are never loaded by the pilot training or tuning loader.
- Rank eligible tuning covers by SHA-256 of canonical JSON containing
  `["pilot_data_v1", dataset_revision, source_identity]`; use the source path to
  break ties. Freeze the first 32. Canonical JSON uses sorted keys, compact
  separators, UTF-8 and one final newline.
- The selection record includes all split counts, selected identities/checksums,
  exclusions, training/tuning checksums and its own checksum. Its checksum covers
  every field except `selection_checksum`. The existing full revision produces
  `9460c45da3dc3e9c5c45170b4c288a135fb470fc045a8cf340c497cbe668104d`.
- CPU fixtures may use smaller revisions. Their GPU profile eligibility is false.
  The exact full revision, required split counts and native tuning coverage are
  required for GPU profile eligibility. Dataset `pilot_ready` stays false.
- Resume saves the shuffled order, consumed position, epoch and independent crop
  random state. Loading a snapshot checks dataset and selection checksums, seed,
  permutation and consumed count. Failed batches restore the previous position.

Run this bounded CPU loading check from the repository root:

```bash
uv run python scripts/measure_pilot_loading.py \
  --dataset-directory datasets/uhd_iqa/75bef6dab2b3de136d1c5c359d6753a25e4a750d5387fe75921e394482f0776c \
  --output-root logs/sprint5_loading --threads 4 --count 16 --deadline-seconds 120
```

It reads at most 16 distinct training images once, writes a timestamped report
under `logs/`, and records shapes, elapsed time, process peak memory and unchanged
file sizes/write timestamps. It runs no optimizer, learned model, image-quality
measurement or held-out image read. The deadline is capped at 120 seconds.
