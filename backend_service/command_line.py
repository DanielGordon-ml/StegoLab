"""Headless configuration and protocol tools sharing application services."""

import argparse
import os
import sys
from pathlib import Path
from typing import Never

from pydantic import ValidationError

from backend_service.failures import ApplicationFailure
from backend_service.protocol_verification import verify_protocol
from backend_service.storage import StateStore
from schemas.base import StrictRecord
from schemas.configuration import ConfigurationProfile

PLANNED_COMMANDS = (
    "encode",
    "decode",
)


class CommandParser(argparse.ArgumentParser):
    """Keep invalid argument values out of command-line error output."""

    def error(self, message: str) -> Never:
        """Replace argparse details with fixed safe guidance."""
        self.print_usage(sys.stderr)
        self.exit(2, "The command arguments are invalid. Use --help for examples.\n")


def build_parser() -> argparse.ArgumentParser:
    """Describe configuration, image, and public protocol verification commands."""
    parser = CommandParser(
        prog="stegolab", description="StegoLab local CPU protocol workspace."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect_configuration", help="Read the saved configuration.")
    validation = commands.add_parser(
        "validate_configuration", help="Validate a JSON configuration without saving."
    )
    validation.add_argument("configuration_file", type=Path)
    inspection = commands.add_parser("inspect_image", help="Validate a cover image.")
    inspection.add_argument("image_file", type=Path)
    preparation = commands.add_parser(
        "prepare_image", help="Prepare an RGB or RGBA PNG without overwriting a file."
    )
    preparation.add_argument("image_file", type=Path)
    preparation.add_argument("output_file", type=Path)
    commands.add_parser(
        "verify_protocol", help="Run bundled public checks; accepts no secret inputs."
    )
    for name, help_text in (
        (
            "prepare_dataset",
            "Prepare a local dataset from request JSON (or - for stdin).",
        ),
        ("inspect_dataset", "Read a dataset summary without checking image integrity."),
        ("validate_dataset", "Verify an offline dataset revision and its images."),
    ):
        dataset = commands.add_parser(name, help=help_text)
        dataset.add_argument("dataset_input")
    for command in PLANNED_COMMANDS:
        commands.add_parser(command, help="Planned; unavailable in this release.")
    from backend_service.model_commands import MODEL_COMMANDS

    for command in MODEL_COMMANDS:
        experimental = commands.add_parser(command, help="Experimental CPU model tool.")
        experimental.add_argument("model_input")
    return parser


def run_command(selected: argparse.Namespace) -> StrictRecord:
    """Dispatch implemented commands explicitly through shared services."""
    if selected.command == "inspect_configuration":
        directory = Path(os.environ.get("STEGOLAB_DATA_DIRECTORY", ".runtime"))
        return StateStore(directory).get_configuration()
    if selected.command == "validate_configuration":
        return ConfigurationProfile.model_validate_json(
            selected.configuration_file.read_text(encoding="utf-8")
        )
    if selected.command == "verify_protocol":
        return verify_protocol()
    from backend_service.image_preparation import inspect_image, prepare_image

    if selected.command == "inspect_image":
        return inspect_image(selected.image_file)
    return prepare_image(selected.image_file, selected.output_file)


def main(arguments: list[str] | None = None) -> int:
    """Print safe metadata on success and fixed errors on expected failures."""
    parser = build_parser()
    selected = parser.parse_args(arguments)
    from backend_service.dataset_commands import (
        DATASET_COMMANDS,
        execute_dataset_command,
    )

    if selected.command in DATASET_COMMANDS:
        return execute_dataset_command(selected.command, selected.dataset_input)
    from backend_service.model_commands import MODEL_COMMANDS, execute_model_command

    if selected.command in MODEL_COMMANDS:
        return execute_model_command(selected.command, selected.model_input)
    if selected.command in PLANNED_COMMANDS:
        print(
            "This command is planned and is not available in this release.",
            file=sys.stderr,
        )
        return 2
    try:
        print(run_command(selected).model_dump_json(indent=2))
        return 0
    except ApplicationFailure as failure:
        print(failure.message, file=sys.stderr)
    except (OSError, UnicodeError, ValidationError):
        if selected.command in ("inspect_configuration", "validate_configuration"):
            message = (
                "The configuration file could not be read or is invalid. Use JSON with "
                "schema version 1 and a positive checkpoint interval divisible by 60."
            )
        else:
            message = "The file could not be processed. Check its format and access."
        print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
