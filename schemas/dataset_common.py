"""Shared bounded paths and counters for offline dataset contracts."""

import re
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, model_validator

from schemas.base import StrictRecord

DatasetSplit = Literal["train", "tuning", "held_out"]
SourceKind = Literal["uhd_iqa", "local", "hugging_face", "https_archive", "upload"]
REMOTE_SOURCE_KINDS: tuple[SourceKind, ...] = (
    "hugging_face",
    "https_archive",
    "upload",
)
SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
DATASET_SPLITS: tuple[DatasetSplit, ...] = ("train", "tuning", "held_out")
SPLIT_MAPPING: dict[str, DatasetSplit] = {
    "training": "train",
    "validation": "tuning",
    "test": "held_out",
}
MAXIMUM_DATASET_ENTRIES = 200_000
MAXIMUM_DATASET_BYTES = 100 * 1024**3


def relative_path(value: str) -> str:
    """Reject paths whose spelling can escape or alias another dataset entry."""
    path = PurePosixPath(value)
    if (
        not value
        or len(value) > 4096
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in value.split("/"))
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("Dataset paths must be safe relative POSIX paths.")
    return value


RelativePath = Annotated[str, AfterValidator(relative_path)]
DatasetName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]


def check_counts(value: dict[str, int]) -> dict[str, int]:
    """Keep user-provided count maps nonnegative and bounded."""
    if any(count < 0 or count > MAXIMUM_DATASET_ENTRIES for count in value.values()):
        raise ValueError("Dataset counts are outside the supported range.")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key) for key in value):
        raise ValueError("Dataset counters must use safe names.")
    return value


CounterMap = Annotated[dict[str, int], AfterValidator(check_counts)]


class DatasetVersionedRecord(StrictRecord):
    """Reject numeric/boolean literal equivalence in strict versioned contracts."""

    @model_validator(mode="before")
    @classmethod
    def validate_literal_types(cls, value: object) -> object:
        """Require literal integers and booleans to retain their exact types."""
        if isinstance(value, dict):
            if "schema_version" in value and type(value["schema_version"]) is not int:
                raise ValueError("Dataset schema version must be an integer.")
            if "pilot_ready" in value and type(value["pilot_ready"]) is not bool:
                raise ValueError("Dataset readiness must be a boolean.")
        return value
