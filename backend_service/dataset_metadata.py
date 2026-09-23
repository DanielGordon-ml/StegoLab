"""Strict local UHD-IQA metadata parsing and source-split preservation."""

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from backend_service.dataset_files import (
    dataset_failure,
    read_source_bytes,
    relative_parts,
)
from backend_service.dataset_scan import SourceInventory
from schemas.dataset_common import SPLIT_MAPPING, DatasetSplit

UpstreamSplit = Literal["training", "validation", "test"]


@dataclass(frozen=True)
class UhdMetadataRow:
    """Retain the upstream image identity and its declared split/subset."""

    image_name: str
    upstream_split: UpstreamSplit
    subset: str
    assigned_split: DatasetSplit
    upstream_identity: str


@dataclass(frozen=True)
class UhdMetadata:
    """Contain frozen source metadata and its exact input byte checksum."""

    rows: dict[str, UhdMetadataRow]
    sha256: str


def read_uhd_metadata(
    root: Path, metadata_relative_path: str, inventory: SourceInventory
) -> UhdMetadata:
    """Validate complete CSV metadata before joining exact image basenames."""
    relative_parts(metadata_relative_path)
    matches = [
        item for item in inventory.files if item.relative_path == metadata_relative_path
    ]
    if len(matches) != 1:
        raise dataset_failure("dataset_metadata")
    data = read_source_bytes(root, matches[0])
    rows: dict[str, UhdMetadataRow] = {}
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), strict=True)
        fields = reader.fieldnames
        if (
            fields is None
            or len(fields) != len(set(fields))
            or not {"image_name", "set", "subset"}.issubset(fields)
        ):
            raise dataset_failure("dataset_metadata")
        for raw in reader:
            name, split, subset = raw["image_name"], raw["set"], raw["subset"]
            if (
                None in raw
                or name is None
                or split is None
                or subset is None
                or len(name) > 255
                or len(subset) > 255
                or len(relative_parts(name)) != 1
                or name in rows
                or split not in SPLIT_MAPPING
                or len(rows) >= 200_000
            ):
                raise dataset_failure("dataset_metadata")
            rows[name] = UhdMetadataRow(
                name,
                cast(UpstreamSplit, split),
                subset,
                SPLIT_MAPPING[split],
                f"uhd_iqa:{name}",
            )
    except (UnicodeError, csv.Error, KeyError, TypeError):
        raise dataset_failure("dataset_metadata") from None
    if not rows:
        raise dataset_failure("dataset_metadata")
    return UhdMetadata(rows, hashlib.sha256(data).hexdigest())
