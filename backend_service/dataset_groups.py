"""Deterministic exact-duplicate and upstream-identity leakage controls."""

from collections import defaultdict

from backend_service.dataset_serialization import canonical_json, checksum
from backend_service.failures import ApplicationFailure
from schemas.dataset_common import (
    DATASET_SPLITS,
    REMOTE_SOURCE_KINDS,
    SPLIT_MAPPING,
    DatasetSplit,
    SourceKind,
)
from schemas.datasets import DatasetImageRecord


def assign_groups(
    records: list[DatasetImageRecord],
    source_kind: SourceKind,
    seed: int = 0,
    *,
    training_intended: bool = True,
    split_mapping: dict[str, DatasetSplit] = SPLIT_MAPPING,
) -> list[DatasetImageRecord]:
    """Group related sources before assigning splits and duplicate representatives."""
    if not records:
        raise ApplicationFailure(
            "dataset_empty",
            "No valid images remain. Add supported JPEG or PNG images.",
            422,
        )
    try:
        return _assign(records, source_kind, seed, training_intended, split_mapping)
    except ValueError:
        raise ApplicationFailure(
            "dataset_groups", "Dataset split or duplicate declarations conflict.", 422
        ) from None


def _check_declarations(
    records: list[DatasetImageRecord], source_kind: SourceKind
) -> None:
    """Apply each source kind's rule for declared splits and identities."""
    declared = [record for record in records if record.upstream_split is not None]
    if source_kind == "local" and declared:
        raise ValueError("Local dataset records require generated splits.")
    if source_kind == "uhd_iqa" and (
        len(declared) != len(records)
        or any(record.source_identity is None for record in records)
    ):
        raise ValueError("UHD-IQA records require source identities and splits.")
    if source_kind in REMOTE_SOURCE_KINDS and (
        declared and len(declared) != len(records)
    ):
        raise ValueError("Remote records declare splits for all images or none.")
    if any(record.source_identity is None for record in declared):
        raise ValueError("Declared splits require source identities.")


def _assign(
    records: list[DatasetImageRecord],
    source_kind: SourceKind,
    seed: int,
    training_intended: bool,
    split_mapping: dict[str, DatasetSplit],
) -> list[DatasetImageRecord]:
    """Use union-find so linked identities and RGB duplicates cannot leak."""
    ordered = sorted(records, key=lambda record: record.source_path)
    if not ordered or len({record.source_path for record in ordered}) != len(ordered):
        raise ValueError("Dataset source paths must be nonempty and unique.")
    _check_declarations(ordered, source_kind)
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        """Find the stable parent while compressing connected identity chains."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    tokens: dict[str, int] = {}
    representatives: dict[str, str] = {}
    for index, record in enumerate(ordered):
        representatives.setdefault(record.rgb_checksum, record.source_path)
        keys = ["rgb:" + record.rgb_checksum]
        if record.source_identity is not None:
            keys.append("identity:" + record.source_identity)
        for key in keys:
            if key in tokens:
                parents[find(index)] = find(tokens[key])
            else:
                tokens[key] = index
    members: dict[int, list[DatasetImageRecord]] = defaultdict(list)
    for index, record in enumerate(ordered):
        members[find(index)].append(record)
    decisions: dict[str, tuple[str, DatasetSplit]] = {}
    for group in members.values():
        identifiers = {"rgb:" + record.rgb_checksum for record in group}
        identifiers.update(
            "identity:" + record.source_identity
            for record in group
            if record.source_identity is not None
        )
        group_id = checksum(canonical_json(sorted(identifiers)))
        labels = {
            record.upstream_split
            for record in group
            if record.upstream_split is not None
        }
        if any(label not in split_mapping for label in labels):
            raise ValueError("A declared split label has no assigned split.")
        declared = {split_mapping[label] for label in labels}
        if len(declared) > 1:
            raise ValueError("A linked group has conflicting official splits.")
        if declared:
            split = next(iter(declared))
        else:
            digest = checksum(canonical_json(["dataset_v1", seed, group_id]))
            bucket = int(digest[:16], 16) % 100
            split = DATASET_SPLITS[0 if bucket < 80 else 1 if bucket < 90 else 2]
        for record in group:
            decisions[record.source_path] = (group_id, split)
    result = [
        record.model_copy(
            update={
                "duplicate_group": record.rgb_checksum,
                "split_group": decisions[record.source_path][0],
                "assigned_split": decisions[record.source_path][1],
                "representative_source_path": representatives[record.rgb_checksum],
            }
        )
        for record in ordered
    ]
    if training_intended and not any(
        record.eligible
        and record.assigned_split == "train"
        and record.representative_source_path == record.source_path
        for record in result
    ):
        raise ApplicationFailure(
            "dataset_training_empty",
            "No eligible unique training images remain. Add training images with "
            "both sides at least 256 pixels.",
            422,
        )
    return result
