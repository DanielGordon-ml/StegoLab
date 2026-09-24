"""Read-only observation of the original command-line proof ownership lock."""

import fcntl
import os
from pathlib import Path


def proof_busy(root: Path) -> bool:
    """Wait for a live external CLI owner without creating or reconciling files."""
    path = root / "state" / "cpu_proof" / ".proof.lock"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        # The worker performs the authoritative check and reports unsafe paths.
        return False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(descriptor)
