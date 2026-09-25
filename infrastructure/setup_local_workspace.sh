#!/bin/sh
# Create local output folders; configure a fresh Linux install for its host owner.
set -eu

stegolab_project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$stegolab_project_root"
mkdir -p datasets checkpoints models state logs data .cache/stegolab/datasets

if [ "$(uname -s)" = Linux ]; then
    if [ "$(id -u)" -eq 0 ] || [ "$(id -g)" -eq 0 ]; then
        echo "Run this setup as the non-root owner of the workspace." >&2
        exit 1
    fi
    touch infrastructure/.env
    if docker volume inspect stegolab_application_data >/dev/null 2>&1 &&
       ! grep -q '^STEGOLAB_USER_ID=' infrastructure/.env; then
        echo "An existing application volume needs its original runtime user IDs." >&2
        echo "Keep its saved IDs or follow the workspace permission guide." >&2
        exit 1
    fi
    if ! grep -q '^STEGOLAB_USER_ID=' infrastructure/.env; then
        printf '\nSTEGOLAB_USER_ID=%s\n' "$(id -u)" >> infrastructure/.env
    fi
    if ! grep -q '^STEGOLAB_GROUP_ID=' infrastructure/.env; then
        printf '\nSTEGOLAB_GROUP_ID=%s\n' "$(id -g)" >> infrastructure/.env
    fi
fi
echo "Workspace folders are ready. Run the Docker Compose start command."
