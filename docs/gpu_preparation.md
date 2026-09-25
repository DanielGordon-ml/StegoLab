# GPU pilot preparation

Sprint 5 prepares an experimental UHD-IQA baseline. It launches no cloud
resources and spends no GPU hours. The CUDA image has a native amd64 CI build
gate; an image definition or CPU test is not GPU acceptance.

## Prepared runtime

- CPU development continues to use the root `pyproject.toml` and `uv.lock`.
- `infrastructure/cuda/` has a separate Linux amd64 / Python 3.12 lock for
  PyTorch `2.14.0+cu126`, CUDA runtime 12.6.3 and cuDNN 9.10.2.21. The
  [official wheel index](https://download.pytorch.org/whl/cu126/torch/) lists
  the CPython 3.12 Linux wheel. Existing common dependencies keep CPU versions.
- Resolve only dependency metadata on Mac. Build/run the image on native
  amd64 CPU CI; do not emulate its CUDA binaries on Apple Silicon.
- Reproduce the dependency record with
  `python -m scripts.pilot_budget_dependencies --check`. It includes the
  independent model runtime and selected CUDA dependency extras.
- A later host needs a compatible NVIDIA driver (conservative minimum
  560.35.05), NVIDIA Container Toolkit and one GPU. See the
  [CUDA 12.6.3 driver table](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html)
  and [NVIDIA container setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Pin the exact tested versions and regional AMI before a paid session.

The separate `Native CUDA image preparation` workflow builds with no GPU
reservation and runs tiny real-model CPU forward, checkpoint continuation and
export checks. It records architecture, image size and available disk. This
does not measure CUDA recovery, GPU memory, throughput, quality or 4K support.

## Required launch inputs — a later operator action

Complete `infrastructure/pilot_launch_inputs.example.json` locally. Every
`REQUIRED` entry and the quota check must be resolved before launch. Keep AWS
credentials in the operator's session; give the application no instance role,
AWS credentials or Docker socket. Do not commit secret input files.

- Use the agreed On-Demand `g6.2xlarge`, Ubuntu 24.04 amd64 GPU base AMI, an
  encrypted 60 GiB root disk and retained encrypted 200 GiB gp3 data disk.
  Do not substitute hardware or choose a region silently.
- Require IMDSv2 with response hop limit 1, termination protection and
  `InstanceInitiatedShutdownBehavior=stop`. Retain the data volume on termination.
- Allow SSH only from the operator's current address. Publish the application
  only on `127.0.0.1:8080`; use an SSH tunnel. Backend port 8000 is not public.
- Record current compute, EBS, snapshot and transfer prices independently.
  Storage charges continue after an instance stops.
- Transfer the already prepared UHD revision; preserve its train/tuning/held-out
  splits. Keep at least 10 GiB or 10% of disk free, including temporary files.
  See [pilot data preparation](pilot_data.md). COCO and the release benchmark
  remain later work; the UHD manifest still has `pilot_ready=false`.

## Persistent accounting and safe stop

Keep **one authoritative operator-side ledger** in a backed-up project directory.
The host receives an exact active-session copy under
`/srv/stegolab/state/gpu_pilot`. The operator copy is necessary because a
stopped instance's data disk cannot accept the final stop confirmation.
Never initialize a second budget for a new host, job, container, or experiment.

All commands below are for a later explicitly approved launch. They do not
launch EC2 themselves. Run them from the repository with the CPU environment:

```sh
uv run --locked python -m scripts.pilot_budget --ledger state/gpu_pilot initialize
uv run --locked python -m scripts.pilot_budget --ledger state/gpu_pilot inspect
```

Before submitting the AWS launch request, record its UTC epoch timestamp on
the operator machine. Register that timestamp as soon as the instance identifier
is known; startup and setup already count. A setup session has at most 7,200
seconds, including the drill, saves and stop. Other allocations are 50,400
baseline, 14,400 ablation and 14,400 final evaluation seconds. The shared
maximum remains 86,400 seconds across every session.

```sh
uv run --locked python -m scripts.pilot_budget --ledger state/gpu_pilot start \
  --session setup_first --instance i-REPLACE --stage setup \
  --started-at REPLACE_WITH_LAUNCH_REQUEST_EPOCH
```

Without `--seconds` the session receives the whole remaining stage allocation
(after the attested lost hour, 3,571 seconds for setup); pass `--seconds` only
to reserve less.

The returned `checkpoint_at` is 300 seconds before the final `deadline_at`;
`poweroff_at` is 120 seconds before it. This leaves 180 seconds for a safe save
and 120 seconds for host shutdown. Create the one-time external EC2-stop
backstop for `poweroff_at` before accepting any GPU work.
`infrastructure/external_stop.example.json` is a template, never an automatic
launch command. Its Scheduler role must trust
`scheduler.amazonaws.com` and allow only `ec2:StopInstances` on this instance;
the operator needs the applicable Scheduler and PassRole permissions. Confirm
the schedule is enabled, has no flexible time window, and targets the correct
instance. Schedule execution is not an exact-second guarantee: retain the
earlier host save guard and verify stopped state externally.

Before copying a ledger, update the host checkout to the same commit that
wrote it: an older checkout rejects newer ledger fields and its guard then
powers the instance off right after boot. Copy the complete active ledger to
the mounted host before work, while the guard and model processes are stopped. Publish the new directory atomically;
do not combine two ledgers or silently replace a different active session.
On every restart, reuse its original deadline. A completed job does not close
the session. A crash leaves the session active until an operator observes EC2
stopped. Accounting charges all overruns rather than capping them at a deadline.

## Host setup and guard

1. Mount the selected EBS data filesystem at `/srv/stegolab` using its recorded
   filesystem UUID in `/etc/fstab`. Check the mount is the intended volume.
   `infrastructure/setup_pilot_host.sh --apply` validates an existing ext4/xfs
   mount and creates the directories; it never formats, launches or trains.
2. Install the exact reviewed repository under `/opt/stegolab`. Create a Python
   3.12 environment at `/opt/stegolab-host`, and install the small hash-locked
   `infrastructure/host/requirements.lock` using `pip install --require-hashes
   --only-binary :all: -r ...`. This host guard does not need Torch.
3. Copy the active ledger with ownership `10001:10001` on the directory and
   every copied file, directory mode 0750 and lock-file mode 0660. Apply these
   permissions only to the inactive host mirror, with all guard/model processes
   stopped; preserve the authoritative operator copy. The root guard preserves
   directory ownership when creating its shared lock. Application containers
   stay non-root. The guard is a separate root host service so a dead
   application cannot disable shutdown.
4. Install `infrastructure/pilot_guard.service` in `/etc/systemd/system/` and
   put the current session identifier in protected
   `/etc/stegolab/pilot-session.env` as `STEGOLAB_SESSION_IDENTIFIER=setup_first`.
   Reload systemd and explicitly start the guard. Check it is active **before**
   model work. Do not enable stale session inputs for a later instance startup.
5. Use the reviewed Compose base plus `infrastructure/compose.gpu.yaml` with
   `STEGOLAB_PERSISTENT_DIRECTORY=/srv/stegolab`. Validate the merged definition
   and pin the built image digest. One explicit CLI container is named
   `stegolab-pilot`; launch the training command as its main process, without
   automatic restart. Pass the already active session identifier to the pilot
   request. The guard sends that container SIGTERM for a safe completed-step
   checkpoint, then calls host `systemctl poweroff` at `poweroff_at`.

The CUDA image installs a locked runtime, not the editable project or its
`stegolab` console script. Use Python module commands inside it. After the
future host guard and external backstop are armed, a reviewed request stored
under the data mount can run as the named CLI container:

```sh
docker compose -f infrastructure/compose.yaml -f infrastructure/compose.gpu.yaml \
  run --rm --no-deps --name stegolab-pilot backend_service \
  python -m backend_service.command_line train /data/pilot_train_request.json
```

Use `python -m backend_service.command_line evaluate REQUEST.json` and
`python -m backend_service.command_line export_models REQUEST.json` for the
other explicit operations. Requests must use schema version 2, CUDA device,
`gpu_pilot` execution mode, output root `/data` and the armed session identifier.
These are future paid-host commands; do not run them during CPU preparation.
Native CPU CI only runs `python -m backend_service.command_line --help` and
the small CPU image checks.

After unexpected host restart, training never resumes automatically. The
external backstop remains active; explicitly re-arm the host guard using the
same session before any manual job resume. The container has no host stop rights.
The application accepts one GPU operation at a time through a separate lock.
Explicit CUDA requests fail when CUDA is unavailable; CPU smoke requests never
start, inspect for execution, or spend a GPU session.

Do not run the executing guard on a development computer. Its default mode
prints simulated process actions; `--execute-host-stop` is only for the armed
root EC2 host service. Tests replace both clocks and subprocess actions.

After the operator observes EC2 state `stopped`, close the authoritative copy:

```sh
uv run --locked python -m scripts.pilot_budget --ledger state/gpu_pilot confirm-stopped \
  --session setup_first --confirmation 'Operator observed EC2 stopped'
```

Use the observation time, even if it is later than physical shutdown. Archive
the operator ledger, AWS state evidence and host logs together. The next launch
uses this same ledger; transfer its new active-session version before model
work. Missing/corrupt accounting requires restoration, not reinitialization.

### Record instance time that ran without a session

If an instance ran while no session was open (as on 2026-09-24, when the host
ran 3,629 seconds during setup), charge that time from independent evidence
such as CloudTrail `RunInstances` and `StopInstances` events. The record is
closed the moment it is written, counts against the named stage, and keeps
the evidence text:

```sh
uv run --locked python -m scripts.pilot_budget --ledger state/gpu_pilot attest-closed \
  --session lost_hour_2026_09_24 --instance i-0ccaee67f0acaa574 --stage setup \
  --started-at 1790283913 --stopped-at 1790287542 \
  --evidence 'CloudTrail RunInstances 2026-09-24T21:05:13Z, StopInstances 22:05:42Z'
```

Attested times must lie in the past, must not overlap a recorded session, and
must fit the stage's remaining allocation. No session may be open at the time.

### Move allocation between stages

Unused seconds can move between stages with a written reason, at most 3,600
seconds per transfer. The 86,400-second total never changes, and a stage can
only give away seconds it has not spent or reserved:

```sh
uv run --locked python -m scripts.pilot_budget --ledger state/gpu_pilot transfer \
  --from-stage ablation --to-stage setup --seconds 3600 \
  --reason 'Readiness drill retry after the lost hour'
```

`inspect` reports every stage's allocation after transfers, its consumed and
its remaining seconds, next to the project totals. Failed commands print a
reason code and a plain explanation instead of a fixed line.

## Backup, restore, and later acceptance

- Take a verified copy of manifests, checkpoints, exported packages, logs,
  runtime identities and ledger after each stopped session. Never keep the
  only copy on ephemeral instance storage.
- Before a live EBS snapshot, stop writes and use SQLite's backup API for a
  consistent database copy. Do not copy a live WAL database alone. For a
  stopped instance, all application writes are already stopped.
- Restore to a separate data volume, verify hashes and ledger totals, then
  validate the restored dataset and checkpoint before allowing explicit resume.
- The later paid readiness drill must prove fresh GPU startup, real CUDA
  forward/backward and export loading, batch memory/throughput, checkpoint
  continuation, private access, a short real stop deadline (more than 300
  seconds), externally confirmed stopped state, and one restore. Its time uses the
  original two-hour setup allocation; no extra free trial budget exists.
- CPU preparation cannot mark these checks passed. This sprint's exit state is
  **preparation verified on CPU; GPU and cloud checks pending**. Critic, LPIPS,
  4K, remote adapters, GUI integration and final model promotion remain open.
