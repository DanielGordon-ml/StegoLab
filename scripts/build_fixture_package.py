"""Build the labelled integer-channel fixture pair as an ordinary package directory."""

import argparse
import sys
import time
from pathlib import Path

from backend_service.failures import ApplicationFailure
from backend_service.model_fixture_channel import (
    FIXTURE_PACKAGE_NAME,
    build_fixture_package,
)


def main(command_arguments: list[str] | None = None) -> int:
    """Create the fixture package for browser tests; never a learned model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=Path("models") / FIXTURE_PACKAGE_NAME,
        help="New package directory; an existing directory is never replaced.",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=300.0,
        help="Stop with a clear failure if packaging takes longer than this.",
    )
    arguments = parser.parse_args(command_arguments)
    if arguments.destination.exists():
        print(
            "The fixture package already exists. Remove it first or choose a new "
            "destination.",
            file=sys.stderr,
        )
        return 2
    try:
        summary = build_fixture_package(
            arguments.destination,
            deadline=time.monotonic() + max(1.0, arguments.deadline_seconds),
        )
    except ApplicationFailure as failure:
        print(failure.message, file=sys.stderr)
        return 1
    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
