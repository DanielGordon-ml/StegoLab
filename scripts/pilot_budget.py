"""Explicit operator commands for persistent EC2 pilot accounting."""

import argparse
import sys
from pathlib import Path
from typing import cast

from backend_service.failures import ApplicationFailure
from backend_service.pilot_budget import confirm_stopped, inspect_budget, start_session
from backend_service.pilot_budget_records import (
    attest_closed_session,
    transfer_allocation,
)
from backend_service.pilot_budget_storage import initialize_pilot_budget
from schemas.pilot_budget import STAGE_SECONDS, PilotStage


def build_parser() -> argparse.ArgumentParser:
    """Describe every accounting command; none of them launches an instance."""
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
    attest = commands.add_parser(
        "attest-closed", help="Record a past run that had no session, from evidence."
    )
    attest.add_argument("--session", required=True)
    attest.add_argument("--instance", required=True)
    attest.add_argument("--stage", choices=STAGE_SECONDS, required=True)
    attest.add_argument("--started-at", type=float, required=True)
    attest.add_argument("--stopped-at", type=float, required=True)
    attest.add_argument("--evidence", required=True)
    transfer = commands.add_parser(
        "transfer", help="Move unused seconds from one stage to another."
    )
    transfer.add_argument("--from-stage", choices=STAGE_SECONDS, required=True)
    transfer.add_argument("--to-stage", choices=STAGE_SECONDS, required=True)
    transfer.add_argument("--seconds", type=float, required=True)
    transfer.add_argument("--reason", required=True)
    return parser


def run(arguments: argparse.Namespace) -> str:
    """Apply one command and return the record it produced as JSON."""
    ledger = arguments.ledger
    if arguments.command == "initialize":
        return initialize_pilot_budget(ledger).model_dump_json(indent=2)
    if arguments.command == "start":
        return start_session(
            ledger,
            arguments.session,
            arguments.instance,
            cast(PilotStage, arguments.stage),
            arguments.started_at,
            requested_seconds=arguments.seconds,
        ).model_dump_json(indent=2)
    if arguments.command == "confirm-stopped":
        return confirm_stopped(
            ledger, arguments.session, arguments.confirmation
        ).model_dump_json(indent=2)
    if arguments.command == "attest-closed":
        return attest_closed_session(
            ledger,
            arguments.session,
            arguments.instance,
            cast(PilotStage, arguments.stage),
            arguments.started_at,
            arguments.stopped_at,
            arguments.evidence,
        ).model_dump_json(indent=2)
    if arguments.command == "transfer":
        return transfer_allocation(
            ledger,
            cast(PilotStage, arguments.from_stage),
            cast(PilotStage, arguments.to_stage),
            arguments.seconds,
            arguments.reason,
        ).model_dump_json(indent=2)
    return inspect_budget(ledger).model_dump_json(indent=2)


def main(command_arguments: list[str] | None = None) -> int:
    """Never launch instances; record existing startup and verified stopped state."""
    arguments = build_parser().parse_args(command_arguments)
    try:
        print(run(arguments))
        return 0
    except ApplicationFailure as failure:
        print(f"{failure.code}: {failure.message}", file=sys.stderr)
    except ValueError as failure:
        print(f"pilot_accounting_invalid: {failure}", file=sys.stderr)
    except OSError:
        print(
            "pilot_accounting_unavailable: the ledger could not be read or written. "
            "Preserve it and check storage access.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
