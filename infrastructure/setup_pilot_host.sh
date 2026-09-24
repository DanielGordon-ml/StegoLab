#!/usr/bin/env bash
# Prepare an already launched host. This script never launches or formats a disk.
set -euo pipefail

if [[ "${1:-}" != "--apply" ]]; then
  printf '%s\n' 'Review docs/gpu_preparation.md first. Use --apply only on the selected EC2 host.'
  exit 0
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]]
[[ "$EUID" == 0 ]]
mountpoint --quiet /srv/stegolab
filesystem_kind="$(findmnt --noheadings --output FSTYPE --target /srv/stegolab)"
[[ "$filesystem_kind" == ext4 || "$filesystem_kind" == xfs ]]
[[ "$(findmnt --noheadings --output TARGET --target /srv/stegolab)" == /srv/stegolab ]]
command -v docker >/dev/null
command -v nvidia-smi >/dev/null
command -v nvidia-ctk >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
docker info >/dev/null

for directory in datasets models checkpoints logs state .cache/stegolab/datasets; do
  install -d -m 0750 -o 10001 -g 10001 "/srv/stegolab/$directory"
done
install -d -m 0750 -o 10001 -g 10001 /srv/stegolab/state/gpu_pilot
install -d -m 0755 /etc/stegolab
printf '%s\n' 'Host directories are ready. No training, guard, or AWS session was started.'
printf '%s\n' 'Install the reviewed host Python environment and systemd guard next; verify the external stop backstop.'
