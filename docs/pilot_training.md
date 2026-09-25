# Pilot preparation commands

Sprint 5 prepares a later GPU readiness test. It spends no GPU hours and does not
approve a model for the application. Existing schema-version-one CPU proof
commands and packages remain supported. Public model capacity is still zero.

## Local checks

```sh
uv run --locked stegolab preflight_pilot docs/examples/pilot_preflight.json
uv run --locked python scripts/verify_gpu_readiness.py --request docs/examples/pilot_readiness.json --check-only
uv run --locked stegolab train docs/examples/pilot_cpu_smoke.json
```

Preflight reports metadata checks, the immutable tuning selection, free space,
runtime versions, and remaining instance budget. Missing GPU accounting is
reported as unused; preflight never initializes it. It does not read held-out
pixels or prove whole-corpus image integrity. Each image is checked against its
recorded checksum when used. GPU and release checks remain `not_run`.

The smoke command performs two engineering-test updates on CPU. It does not
select a model, measure image quality, or modify the dataset. Its outputs live
under `.runtime/pilot_smoke/`. Repeating that experiment requires an explicit
checkpoint resume or a new experiment identifier. CPU smoke is capped at ten
absolute optimizer steps and a requested allowance no greater than 120 seconds.
Work stops at completed-step boundaries with up to ten seconds reserved for
saving; a slow final step or atomic save can finish after the nominal allowance.
It uses neither the Sprint 4 proof ledger nor the GPU ledger.

## Frozen training behavior

- FP32 dense encoder/decoder, 256-pixel random crops, Adam at `1e-4`.
- Physical batch four or eight; accumulation consumes exactly sixteen samples.
- Synchronous shuffled loading; no prefetched samples are counted as consumed.
- Image-loss weight rises from zero to 100 over `ceil(planned_steps / 5)` steps.
- The step count, physical batch and schedule are frozen in the checkpoint.
  The later GPU probe informs a new run's settings; resume never recalculates them.
- Complete state includes BatchNorm, Adam, global random states, sample order,
  crop randomness, and code/data/environment identities.
- Resume requires the same physical batch, device and numerical environment.
  Different physical batches need not produce equal results because BatchNorm
  observes different groups of images.
- Checkpoints retain latest recovery, three best measured candidates and pinned
  states. CPU smoke has no measured candidates. Deployment exports are separate.

GPU training requires `execution_mode: "gpu_pilot"`, `configuration.device:
"cuda"`, an existing `session_identifier`, and the frozen full UHD-IQA profile.
CUDA must expose exactly one device. Missing CUDA never falls back to CPU.
No command launches EC2 or opens a paid session. See [GPU preparation](gpu_preparation.md)
for explicit operator setup and the independent deadline guard.

## Resume, evaluation and export

The report's `checkpoint` path is relative to `output_root`. For input requests,
join those two paths (or use an absolute path). For example, the smoke checkpoint
is `.runtime/pilot_smoke/checkpoints/pilot_cpu_smoke/<checkpoint identifier>`.

Copy the original training JSON, add `resume_checkpoint` pointing to the returned
checkpoint directory, and increase `stop_after_step` only within the original
`planned_optimizer_steps`. Preserve the experiment identifier and configuration.
The model from Sprint 4 is not a pilot resume checkpoint; begin a new pilot run.

Schema-version-two evaluation requests use the following fields:

```json
{
  "schema_version": 2,
  "experiment_identifier": "pilot_cpu_smoke",
  "output_root": ".runtime/pilot_smoke",
  "execution_mode": "cpu_smoke",
  "device": "cpu",
  "checkpoint": "<returned checkpoint directory>",
  "dataset_directory": "<the original prepared revision>"
}
```

Pass that JSON to `stegolab evaluate`. CPU smoke checks one full-message PNG
without quality ranking. GPU evaluation processes the 32 frozen native tuning
covers using full 1,024-byte public messages. Every recovery and operational
failure remains in the report. Deadline stops are incomplete, never passes.
Tuning evidence does not qualify a release or establish held-out generalization.

For `stegolab export_models`, use the same common fields, remove
`dataset_directory`, and add `export_name`, such as `002V`. Exports use frozen
CPU copies of the weights and remain experimental. Existing destinations are
never overwritten. A CPU export does not prove CUDA compatibility.

On the later paid host, an export request with `device: "cuda"` requires its
active GPU session. Publication first checks CPU parity, then actual isolated
CUDA parity. Both must pass before publication; each verification records its
device. Standalone CUDA runtimes use the same deterministic FP32 settings as
training. The [GPU readiness guide](gpu_readiness.md) explains the separate
authenticated-message and resume checks.

Version-two packages contain separate `requirements-cpu.txt` and
`requirements-cuda.txt` plus dependency records in the manifest. Install only the
selected runtime's requirements. Each package runs outside the repository:

```sh
python runtime.py verify --device cpu
python runtime.py tensor --device cpu --image image.npy --output result.npy
```

An encoder tensor command also takes `--payload payload.npy`. The future CUDA
environment uses `--device cuda`; version-one packages remain CPU-only. Text
commands still read secrets from protected stdin and use authenticated recovery.
An encoder package needs no decoder weights; application download verification
requires the matching separate decoder.

## Optional critic and forking (built, not yet run)

Sprint 6 adds two tools for the later GPU ablation. Neither has been used in a
real training run yet, and neither changes the exported packages.

- **Critic.** `configuration.critic` is off by default. With
  `{"enabled": true, "weight": 1.0}` a small realism critic (three convolution
  stages, one score per image) trains alongside the pair: it learns to score
  original covers above encoded images, and the encoder receives an extra loss
  term that pushes its images towards "real". The critic uses one encoder
  forward per batch, its own Adam optimizer (`learning_rate`, default 0.0001)
  and weight clipping (`weight_clip`, default 0.1). A weight of zero leaves the
  encoder and decoder bit-identical to a run without the critic. Every step
  record then carries `critic_loss`. Checkpoints with a critic use state layout
  version two. The loader still reads layout version one, so earlier
  checkpoints still evaluate and export; resume and fork also require the code
  identity to match, so pilot checkpoints saved before this change can only be
  evaluated and exported, not continued.
- **Fork.** `fork_from_checkpoint` copies the complete training state of an
  existing pilot checkpoint (weights, optimizer, sampler position and random
  state) into a **new** experiment identifier, so an ablation can start from
  the baseline's pinned checkpoint. The dataset, selection, environment, code
  and frozen settings must match; only the critic block may differ. The new
  checkpoints record `forked_from` as `<experiment>:<checkpoint identifier>`.
  A fork cannot reuse the source experiment's identifier, and a request cannot
  resume and fork at the same time.

## Evidence boundaries

The readiness report keeps `passed_locally`, `failed`, and `not_run` distinct.
The [data guide](pilot_data.md) records exact selection and loading measurements.
The [Sprint 5 record](../plan/sprint_05.md) records acceptance and remaining gates.
No COCO benchmark, near-duplicate review, critic training run, 4K, GUI model
operation or new model-quality experiment is included in this sprint; the critic
and fork exist as code only.
