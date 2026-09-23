"""Small headless commands using the same settings schema and store as the API."""

import argparse
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from backend_service.failures import ApplicationFailure
from backend_service.storage import StateStore
from schemas.configuration import ConfigurationProfile

PLANNED_COMMANDS = (
    "prepare_dataset",
    "train",
    "evaluate",
    "encode",
    "decode",
    "export_models",
    "inspect_checkpoint",
)


def build_parser() -> argparse.ArgumentParser:
    """Describe implemented configuration commands and planned model commands."""
    parser = argparse.ArgumentParser(
        prog="stegolab", description="StegoLab local CPU application foundation."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect_configuration", help="Read the saved configuration.")
    validation = commands.add_parser(
        "validate_configuration", help="Validate a JSON configuration without saving."
    )
    validation.add_argument("configuration_file", type=Path)
    for command in PLANNED_COMMANDS:
        commands.add_parser(command, help="Planned; unavailable in this release.")
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Run safe settings inspection or validation and return an exit code."""
    parser = build_parser()
    selected = parser.parse_args(arguments)
    if selected.command in PLANNED_COMMANDS:
        print(
            "This command is planned and is not available in this release.",
            file=sys.stderr,
        )
        return 2
    try:
        if selected.command == "inspect_configuration":
            directory = Path(os.environ.get("STEGOLAB_DATA_DIRECTORY", ".runtime"))
            configuration = StateStore(directory).get_configuration()
        else:
            configuration = ConfigurationProfile.model_validate_json(
                selected.configuration_file.read_text(encoding="utf-8")
            )
        print(configuration.model_dump_json(indent=2))
        return 0
    except ApplicationFailure as failure:
        print(failure.message, file=sys.stderr)
    except (OSError, UnicodeError, ValidationError):
        print(
            "The configuration file could not be read or is invalid. Use JSON with "
            "schema version 1 and a positive checkpoint interval divisible by 60.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
