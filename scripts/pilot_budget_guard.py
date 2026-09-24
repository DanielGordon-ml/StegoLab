"""Host-only stop guard; dry-run prints actions and never signals or powers off."""

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from backend_service.pilot_guard import run_guard


@dataclass
class HostStopActions:
    """Send one container a save signal, then stop the EC2 host independently."""

    container_name: str
    execute: bool = False

    def _run(self, command: list[str]) -> None:
        """Bound host commands; simulation never invokes a process."""
        if not self.execute:
            print("simulated_action: " + " ".join(command), flush=True)
            return
        try:
            subprocess.run(command, check=True, timeout=10, capture_output=True)
        except (subprocess.SubprocessError, OSError):
            raise OSError(
                "The host stop action failed; use the external backstop."
            ) from None

    def request_checkpoint(self) -> None:
        """Ask the CLI container to save after its current complete optimizer step."""
        self._run(["docker", "kill", "--signal=TERM", self.container_name])

    def poweroff(self) -> None:
        """Power off rather than halt; the EC2 launch setting must select stop."""
        self._run(["systemctl", "poweroff"])


def main() -> int:
    """Require a deliberate host-only execute flag; the default is simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--container", default="stegolab-pilot")
    parser.add_argument("--execute-host-stop", action="store_true")
    arguments = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", arguments.container):
        parser.error("Choose a valid container name.")
    if arguments.execute_host_stop and os.geteuid() != 0:
        parser.error("Executing the host guard requires the root host service.")
    actions = HostStopActions(arguments.container, arguments.execute_host_stop)
    try:
        run_guard(arguments.ledger, arguments.session, actions)
        return 0
    except OSError:
        print("Host stop failed. Confirm the external EC2 stop backstop is active.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
