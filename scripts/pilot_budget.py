"""Explicit operator commands for persistent EC2 pilot accounting."""

import argparse
from pathlib import Path
from typing import cast

from backend_service.failures import ApplicationFailure
from backend_service.pilot_budget import confirm_stopped, inspect_budget, start_session
from backend_service.pilot_budget_storage import initialize_pilot_budget
from schemas.pilot_budget import STAGE_SECONDS, PilotStage


def main() -> int:
    """Never launch instances; record existing startup and verified stopped state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("initialize")
    commands.add_parser("inspect")
    start = commands.add_parser("start")
    start.add_argument("--session", required=True)
    start.add_argument("--instance", required=True)
    start.add_argument("--stage", choices=STAGE_SECONDS, required=True)
    start.add_argument("--started-at", type=float, required=True)
    start.add_argument("--seconds", type=float)
    stop = commands.add_parser("confirm-stopped")
    stop.add_argument("--session", required=True)
    stop.add_argument("--confirmation", required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "initialize":
            result = initialize_pilot_budget(arguments.ledger)
            print(result.model_dump_json(indent=2))
        elif arguments.command == "start":
            session = start_session(
                arguments.ledger,
                arguments.session,
                arguments.instance,
                cast(PilotStage, arguments.stage),
                arguments.started_at,
                requested_seconds=arguments.seconds,
            )
            print(session.model_dump_json(indent=2))
        elif arguments.command == "confirm-stopped":
            ledger = confirm_stopped(
                arguments.ledger, arguments.session, arguments.confirmation
            )
            print(ledger.model_dump_json(indent=2))
        else:
            print(inspect_budget(arguments.ledger).model_dump_json(indent=2))
        return 0
    except (ApplicationFailure, OSError, ValueError):
        print("Pilot accounting failed. Preserve the ledger and check its state.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
