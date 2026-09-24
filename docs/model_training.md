# Experimental CPU model workflow

For Sprint 5's separate version-two pilot commands, larger-data loader, and
explicit CPU/CUDA packages, use the [pilot preparation guide](pilot_training.md).
The version-one proof below retains its original settings and budget.

Sprint 4 adds a small command-line learning proof. The browser still has no
installed model or usable payload capacity. A completed training command means
the requested steps finished; it does not mean the learning gate passed.

## Install and prepare

- Use Python 3.12 and `uv sync --locked`. PyTorch is pinned to 2.14.0 from its
  explicit CPU package index; all other packages use the normal package index.
- The same locked dependencies build on native macOS arm64 and Linux CPU.
- Start from the existing frozen 30-image UHD-IQA sample. The example request
  names its exact revision. Run commands from the repository root.
- Each invocation validates the prepared revision once. Training uses the first
  four eligible representatives in the training split and the first four in
  tuning. It never scores or learns from the held-out split.
- Four fixed center crops per split are cached at 256×256. Larger tuning crops
  are loaded and checksum-checked when needed. No source image is modified.

```sh
uv run --locked stegolab train docs/examples/cpu_proof_train.json
```

The fixed development profile uses CPU float32, four physical batches of one
image per optimizer step, 32 hidden channels, dense connections, and a residual
encoder. Each physical batch receives fresh random bits. BatchNorm sees a batch
of one; accumulation does not change its statistics. The image loss weight is
`100 * min(completed_optimizer_steps / 200, 1)`. Both losses use mean reduction.
Training uses rounding with a straight-through gradient; evaluation uses the
actual 8-bit PNG values. Validation uses independent fixed random generators.

## Requests and saved state

All model requests accept a JSON file or `-` for standard input. Requests reject
unknown fields; never include a password or message in a training request.
The training example fixes `experiment_identifier`, `output_root`, the dataset
revision directory, and `stop_after_step`. Only CPU thread count is adjustable
inside the fixed training configuration; other baseline choices are frozen.

- `stop_after_step` is an absolute boundary, at most 1,000. A smaller value is
  useful for a controlled stop; it does not change the loss schedule.
- Ctrl-C or SIGTERM requests a save after the current complete optimizer step.
  The command returns 130 or 143 after a successful signal-triggered save.
- Resolve the returned checkpoint path relative to `output_root`, then put that
  full path into `resume_checkpoint`. Keep the same
  experiment, dataset, output root, settings, and numerical environment, and
  increase `stop_after_step` to continue. No training restarts automatically.
- If startup fails before any checkpoint exists, the corrected request can
  reuse that experiment identifier. Its spent time still counts. Existing or
  unclear checkpoint state requires explicit resume or repair.
- Checkpoints save model and BatchNorm weights, Adam state, Python/NumPy/Torch
  random states, sampling position, configuration, data revision, and numerical
  environment. They contain no plaintext or passwords.

```sh
uv run --locked stegolab inspect_checkpoint checkpoints/EXPERIMENT/CHECKPOINT
```

Run files live under the chosen output root:

| Location | Contents |
|---|---|
| `state/cpu_proof/` | Locked, persistent time ledger |
| `checkpoints/<experiment>/` | Atomic checkpoint directories and retention index |
| `logs/<UTC date>_<experiment>_<unique>/` | Step metrics, validation and command reports |
| `models/<export_name>/` | Separate encoder and decoder deployment packages |

Recovery checkpoints are saved every five minutes, after validation, and on a
normal stop. Retention keeps the latest, three best measured candidates, and
pinned checkpoints. Ranking uses exact recovery, PSNR, then SSIM. The latest
checkpoint may be unmeasured; select a measured candidate from `index.json` for
the full proof. Export packages are never pruned by checkpoint cleanup.

## Evaluate a frozen checkpoint

Create a request with these fields, replacing the checkpoint directory:

```json
{
  "experiment_identifier": "sprint04_baseline",
  "output_root": ".",
  "checkpoint": "checkpoints/sprint04_baseline/CHECKPOINT",
  "dataset_directory": "datasets/uhd_iqa/c3b9eb1bea77b4172a620814f047e5214d89b24ef53639582980546cfb20d0d2"
}
```

```sh
uv run --locked stegolab evaluate evaluation_request.json
```

- Full evaluation scores 16 fixed random-map cases and 36 saved/reopened PNG
  trials: four tuning covers × 512×512, 513×517, 1024×1024 × empty, Unicode,
  and full-capacity public messages. The final size includes 1,024-byte messages.
- The learning gate needs bit error rate at most 10% and all 36 exact
  authenticated recoveries. Every attempt is reported; failed inputs are not
  replaced or retried until successful.
- `lightweight: true` runs four maximum-message 512×512 PNG trials, matching
  periodic training validation. It cannot pass the full learning gate.
- Reports include PSNR, mean RGB SSIM, clipping, time, and process peak memory.
  SSIM uses an 11-pixel Gaussian window, sigma 1.5, valid borders, and range 1.
  Identical pixels use `identical_pixels: true` and null PSNR, rather than invalid
  JSON infinity. Lifetime peak memory includes earlier work in that process.
- A completed scientific failure returns a complete report with a false gate.
  Incomplete or failed evaluation returns exit code 1 and preserves its report.

## Export independent packages

```json
{
  "experiment_identifier": "sprint04_baseline",
  "output_root": ".",
  "checkpoint": "checkpoints/sprint04_baseline/CHECKPOINT",
  "export_name": "001V"
}
```

```sh
uv run --locked stegolab export_models export_request.json
python models/001V/encoder/runtime.py --help
python models/001V/decoder/runtime.py --help
```

Each package has only its own network, a tensor runtime, the required image and
protocol helpers, a manifest, checksums, pinned CPU dependencies, and a public
known-answer vector. The decoder needs no original cover or encoder package.
Encryption and PNG handling are outside the exported tensor graph. The graph
accepts one float32 image with sides from 512 to 1024; a matching encoder payload
has one channel. These experimental shape bounds do not qualify every shape for
production. Keep the exact runtime version recorded in the package.

Publication first runs six isolated eager/export comparisons at all three
proof sizes. The pair root contains `verification.json`; a failed check leaves
no published pair. To check the complete public-message matrix against the
actual packages, use the same experiment ledger:

```sh
uv run --locked python scripts/verify_model_packages.py \
  --package models/001V \
  --dataset datasets/uhd_iqa/c3b9eb1bea77b4172a620814f047e5214d89b24ef53639582980546cfb20d0d2 \
  --experiment sprint04_baseline
```

Add `--container-image <local CPU backend image>` for the Linux check. The host
keeps the shared time ledger; the container uses no network, read-only inputs,
a non-root user, 10 GiB memory, and a 64 MiB temporary filesystem. Each encoder
and decoder invocation runs in a fresh isolated process. The check writes a
safe JSON report and returns failure if the complete recovery gate is not met.

Standalone encoding accepts message/password through bounded JSON standard
input; decoding accepts the password the same way. Never put real secrets in
command arguments, environment variables, request files, or logs. The encoder
package cannot perform decoder verification on its own. A successful standalone
encode is not application approval to distribute a PNG. The application must
still verify exact authenticated recovery before offering a download.

## Budget and limits

- One initial experiment and at most one diagnostic rerun share a four-hour
  wall-clock budget, not four core-hours. Each experiment gets at most two hours
  including data checks, validation, saving, and exports.
- Training stops with twenty minutes reserved for saving and proof checks.
  Resume and separate evaluation/export commands consume the same saved budget.
- The ledger holds an exclusive lock. An unclean process exit conservatively
  charges its reserved remaining allowance. Never delete or replace the ledger
  to extend an experiment. Keep it when reporting incomplete results.
- Atomic writes preserve earlier completed checkpoints on failure. The writer
  requires 10 GiB or 10% free space, whichever is larger.
- CPU learning may fail within the fixed budget. Its gate stays open. This
  small sample cannot establish release quality, detection resistance, or SOTA.
- Remote data, GUI training, worker scheduling, fine-tuning, attacks, critics,
  tiling, 4K inference, and GPU runs remain later work.

See [Sprint 4](../plan/sprint_04.md) for measured acceptance and limitations.
