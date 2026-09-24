# StegoLab Infrastructure Plan

Status: researched implementation plan; no AWS resources have been launched.

Sprint 5 implements CUDA build preparation, the instance-time ledger, mocked
deadline controls, and a manual host runbook. See [GPU preparation](../docs/gpu_preparation.md)
and [acceptance](sprint_05.md). No EC2 launch, physical stop drill or GPU test is
implied by these files; those remain later explicit gates.
Research date: 2026-09-23. Related plans: [Backend](backend.md), [Frontend](frontend.md), [Execution order](execution_order.md).

## 1. Deployment shape

- Keep infrastructure definitions and host scripts in infrastructure/. Use Docker Compose for the first release.
- Run two application containers: frontend/reverse proxy and backend. The backend supervisor runs the API plus separate spawned GPU and dataset-download worker processes.
- Use bounded in-memory process channels for secret inputs and commands. The API/scheduler owns persistent job state. Do not add Redis, Kubernetes, a distributed scheduler, or separate accounts to this release.
- One GPU operation runs at a time. Training/evaluation occupies that slot; inference queues. Worker heartbeat failure releases stale ownership only after the supervisor confirms the old process has stopped.
- After restart, reconcile jobs before accepting new work. Training requires explicit checkpoint resume; secret-bearing inference becomes needs_input. Never restart paid training automatically.
- The [Archify map](../docs/architecture.md) shows the browser, frontend container, backend API, implemented local dataset services, and planned workers/models/cache. Solid and dashed paths distinguish implemented work from planned components; independent deployment packages remain planned.

## 2. Local and cloud environments

| Environment | Choice |
|---|---|
| Mac development | Native Linux arm64 CPU containers; tiny model tests and application work. |
| Full training | Linux amd64 CUDA containers on one NVIDIA EC2 instance. |
| Default EC2 instance | On-Demand g6.2xlarge: one L4 GPU, 8 virtual CPUs, 32 GiB system RAM. |
| Host image | AWS Deep Learning Base GPU image for Ubuntu 24.04, resolved in the chosen region and pinned by image identifier. |
| Persistent storage | Encrypted 60 GiB root volume and separate encrypted 200 GiB gp3 data volume mounted at /srv/stegolab. |

- AWS lists 24 GB GPU memory for G6; preflight uses actual driver-reported memory and keeps headroom. [AWS G6 specifications](https://aws.amazon.com/ec2/instance-types/g6/)
- Use native architecture builds; do not emulate the CUDA image on Apple Silicon. [Docker multi-platform guidance](https://docs.docker.com/build/building/multi-platform/)
- Resolve and record the host image, driver, PyTorch/CUDA versions, image digests, and dependency locks as one tested set. [AWS base-image release notes](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpubasesinglecuda-ul2404-2026-06-05.html)
- Build separate CPU and CUDA backend targets from the same application code. Validate torch.export loading in both before full training.
- Use a manual EC2 launch runbook followed by an idempotent host setup script. Region, subnet, quota, image identifier, current hourly rate, and storage cost are required launch inputs; do not invent a regional price or silently choose another GPU.
- Install/configure NVIDIA Container Toolkit if absent and verify GPU access inside the container. Reserve exactly one GPU in the CUDA Compose override. [NVIDIA setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), [Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)
- Initial aggregate backend RAM limits: 10 GiB on Mac and 24 GiB on EC2; frontend 256 MiB. Leave host headroom and start with small physical batches. These are host-memory limits, not GPU-memory guarantees.
- Run containers as non-root, without privileged mode or a mounted Docker socket. Use bounded log rotation, health checks, and a grace period that allows checkpoint writes.

## 3. Private access and credentials

- Bind the frontend/proxy only to 127.0.0.1:8080. Backend and database have no published host ports.
- Reverse-proxy /api/v1 to the backend and disable buffering for SSE. Bundle all frontend assets locally.
- Open the remote app through an SSH tunnel. Restrict EC2 SSH ingress to the operator's current address; expose no public application port.
- Use current Docker Engine and verify that localhost publication is not externally reachable. Older engines had a documented localhost-port limitation. [Docker port publication](https://docs.docker.com/engine/network/port-publishing/)
- The application receives no AWS credentials or instance role. The operator performs launch, snapshot, stop, and restore operations using their AWS session.
- Require IMDSv2 and a metadata response hop limit of 1. Download code must still reject metadata/link-local destinations and validate resolved addresses/redirects. Metadata settings alone are not a downloader sandbox. [AWS metadata options](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-instance-metadata-options.html)
- Mount optional Hugging Face read tokens as protected secret files. Compose secrets do not encrypt the source file automatically. Exclude these files from Git and configuration exports. [Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/)
- Never persist message passwords or plaintext in process arguments, environment variables, logs, telemetry, job metadata, or browser storage.
- Because the first release uses one backend container, process separation is for responsiveness and recovery, not a hostile-code security boundary. Dataset imports must never execute downloaded scripts.

## 4. Data, disk limits, and recovery

- Keep prepared datasets, cache, deployment models, checkpoints, logs, and SQLite state on the persistent data volume.
- Runtime layout: datasets/; .cache/stegolab/datasets/; models/; checkpoints/; logs/<run_date>/<run_identifier>/; state/.
- Keep the data volume on termination and enable termination protection for the pilot. Do not rely on instance-store NVMe for the only copy of data: its contents are lost on stop. [AWS stop/start behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html)
- Maintain at least 10 GiB or 10% free space, whichever is larger. Preflight includes downloaded archives, extraction, prepared data, checkpoints, and temporary writes.
- Write checkpoints and exports to temporary files on the same filesystem, flush, rename atomically, and publish metadata only after successful verification. Keep the previous valid checkpoint until replacement succeeds.
- Retain latest recovery, three best, and pinned checkpoints. Never prune deployment packages through checkpoint cleanup.
- SQLite lives on the local filesystem with WAL and short transactions. Do not use a shared network filesystem or copy a live database file without its backup procedure. [SQLite WAL](https://www.sqlite.org/wal.html)
- Take a consistent SQLite backup and pause file writes before EBS snapshots. Snapshot after completed pilot stages and before upgrades. Perform one restore test. [EBS snapshot guidance](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-creating-snapshot.html)
- Inference uploads/results expire after 24 hours or explicit deletion. Download cancellation removes only that job's partial assets. Cache deletion must not break prepared datasets or active jobs.
- Log files contain descriptive structured events, versions, durations, resource measurements, and diagnostic references; they exclude messages, passwords, tokens, and signed URL query strings.

## 5. Download controls

- HTTPS only for remote assets; reject embedded credentials, unexpected ports, local/private/reserved addresses, and metadata endpoints.
- Resolve and validate IPv4/IPv6 destinations at connection time and on every redirect, including DNS-rebinding protection.
- Bound total bytes, extracted bytes, file counts, timeouts, and redirects. Reject archive traversal, escaping symlinks, and decompression bombs.
- Keep Hugging Face transport/cache operations separate from general URL imports. No automatic credential forwarding to unrelated hosts.
- Show unknown download sizes honestly. Do not promise a universal three-second stop-and-cleanup deadline.
- Tests must cover a normal dataset, redirect chains, blocked addresses, corrupt archives, cancellation, offline reuse, and disk exhaustion.

## 6. Pilot budget and shutdown

- Use one persistent project-wide ledger of 24 allocated GPU-instance hours across sessions. Count startup, setup, idle time, training, evaluation, saving, and shutdown overhead; log active training time separately.
- Reconcile the previous session after crashes or stop/start. Restarting a job or container must not reset the budget.
- Each session has a deadline derived from the remaining ledger. Stop accepting work when insufficient time remains; begin checkpointing before the deadline.
- A host-level deadline guard works independently of the API, requests a safe save, and powers off with InstanceInitiatedShutdownBehavior=stop. Do not use halt. [AWS shutdown behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html)
- For unattended sessions, use an external one-time EC2-stop backstop through EventBridge Scheduler with permissions limited to this instance. Confirm stopped state after the session. [Scheduler targets](https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-targets-universal.html)
- Track EBS, snapshots, network, and address charges separately: stopping the instance does not remove storage charges. [AWS stop/start costs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-stop-start-works.html)

## 7. CI, release, and acceptance

- Pull-request CI: formatting/lint/type checks without redundant tests, meaningful protocol/API tests, frontend tests, one small browser workflow, and CPU image build.
- Use dependency locks, full commit hashes for third-party actions, read-only repository permissions by default, and no cloud credentials in pull-request jobs. [GitHub secure workflows](https://docs.github.com/en/actions/reference/security/secure-use)
- Keep long training and 10,000-image evaluation out of routine pull-request CI. Actual GPU execution is an explicit EC2 validation stage.
- Acceptance checks: fresh Mac install; fresh EC2 GPU start; exported encoder/decoder smoke tests; browser access through SSH; failed external application-port probe; SSE streaming; container and host restart; interrupted checkpoint; disk-full recovery; download controls; consistent restore.
- Run a short-budget drill before the full pilot to prove that time accounting persists and the stop guard works.
- Record image digests, resolved configuration, data manifests, code commit, model checksums, and evaluation report together. Rollback selects an existing known-good image/model pair rather than overwriting it.
- An infrastructure pass does not establish message recovery, stealth, or SOTA; those remain separate backend gates.
