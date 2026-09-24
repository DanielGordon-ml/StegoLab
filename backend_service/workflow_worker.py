"""Run an existing command with parent-loss shutdown and bounded inputs."""

import os
import signal
import sys
import threading
import time

from backend_service.command_line import main


def watch_parent(parent_identifier: int) -> None:
    """Request shutdown when the owning API process disappears."""
    while os.getppid() == parent_identifier:
        time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGTERM)


def run() -> int:
    """Keep execution in a fresh process using the established command services."""
    parent = int(os.environ.get("STEGOLAB_WORKER_PARENT_IDENTIFIER", os.getppid()))
    threading.Thread(target=watch_parent, args=(parent,), daemon=True).start()
    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(run())
