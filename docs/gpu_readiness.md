# Later GPU readiness check

- Sprint 5 prepares this script. No GPU run is required or recorded locally.
- `--check-only` is the default. It checks frozen data metadata and writes a report
  under the selected output root's `logs/` folder. Every CUDA check stays
  `not_run`. It does not call CUDA, open a host session or start its ledger.
- `--run` requires an existing active host session in `state/gpu_pilot`. The
  command holds the exclusive operation lease and obeys its work deadline.
  It never starts EC2, changes the 24-hour allowance or starts a baseline run.

Prepare a JSON request using the actual dataset, persistent output root and
existing host session:

```json
{
  "schema_version": 2,
  "dataset_directory": "/persistent/datasets/uhd_iqa/75bef6dab2b3de136d1c5c359d6753a25e4a750d5387fe75921e394482f0776c",
  "output_root": "/persistent/stegolab",
  "experiment_identifier": "gpu_readiness",
  "session_identifier": "setup_first",
  "learned_checkpoint": "/persistent/checkpoints/frozen_learned_checkpoint",
  "learned_experiment_identifier": "original_learned_experiment"
}
```

```bash
uv run python scripts/verify_gpu_readiness.py --request readiness.json --check-only
# Only on the later authorized GPU host, inside its existing setup session:
python scripts/verify_gpu_readiness.py --request readiness.json --run
```

- The real probe requires exactly one visible GPU and the locked PyTorch 2.14.0
  CUDA 12.6 environment. It performs a device arithmetic check, then compares
  continuous and interrupted training separately for physical batches 4 and 8.
- Each comparison uses two optimizer updates, effective batch 16, and a fresh
  readiness experiment. It checks model and optimizer tensors, Python/NumPy/Torch
  random state, sample order, and the next payload bits and crop. Reports include
  training time and peak device memory to support a later baseline choice.
- Optional learned weights are read as frozen CPU weights from either a legacy
  proof checkpoint or a pilot checkpoint. Separate version-two packages are
  created under that readiness run's logs. Both package graphs are checked in
  isolated CUDA processes against eager tensors, including odd image sizes.
- The saved/reopened PNG check attempts all 32 frozen tuning covers once, each
  with a full 1,024-byte authenticated message. Every attempt is recorded. A
  recovery failure or incomplete deadline is a failed check; it is not retried
  and the gate is not reduced.
- Omit both learned checkpoint fields if none is available. Export and message
  recovery remain `not_run`, and `readiness_passed` remains false. A short random
  training probe is not treated as evidence of learned message recovery.
- The command exits nonzero for any failure or incomplete explicit GPU run. Its
  reports leave dataset `pilot_ready` false. Shutdown evidence, baseline step
  selection and release-quality experiments remain separate gates.
