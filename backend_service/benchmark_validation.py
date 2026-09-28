"""Verify frozen release-benchmark identities without changing the folder."""

import logging
from pathlib import Path
from typing import Literal

from backend_service.benchmark_freeze import (
    IDENTITIES_FILE_NAME,
    MEMBERS_FILE_NAME,
    benchmark_failure,
    derive_benchmark,
    identities_checksum,
    members_checksum,
    selection_key,
)
from backend_service.dataset_serialization import (
    canonical_json,
    contained_file,
    read_bounded,
    read_records,
)
from backend_service.failures import ApplicationFailure
from schemas.benchmark_identities import (
    BenchmarkMember,
    BenchmarkValidationReport,
    MemberRole,
    ReleaseBenchmarkIdentities,
)

logger = logging.getLogger(__name__)

MAXIMUM_IDENTITIES_BYTES = 1024 * 1024
Integrity = Literal["verified", "verified_with_annotations"]


def _validation_members_ordered(
    header: ReleaseBenchmarkIdentities, members: list[BenchmarkMember]
) -> bool:
    """Tell whether the leading members are validation members sorted by path."""
    validation = members[: header.validation_count]
    paths = [member.member_path for member in validation]
    return all(member.role == "validation" for member in validation) and (
        paths == sorted(paths)
    )


def _ranked_members_ordered(
    header: ReleaseBenchmarkIdentities, members: list[BenchmarkMember]
) -> bool:
    """Tell whether the trailing members carry contiguous ranks in key order."""
    ranked = members[header.validation_count :]
    rule = header.selection_rule
    expected_roles: list[MemberRole] = [
        "training_source" if rank < header.training_source_count else "reserve"
        for rank in range(header.training_source_count + header.reserve_count)
    ]
    if [member.rank for member in ranked] != list(range(len(expected_roles))):
        return False
    if [member.role for member in ranked] != expected_roles:
        return False
    keyed = sorted(
        ranked,
        key=lambda member: (
            selection_key(rule.policy_version, rule.seed, member.member_path),
            member.member_path,
        ),
    )
    ordered = [member.member_path for member in ranked]
    return ordered == [member.member_path for member in keyed] and (
        rule.population_count >= len(ranked)
    )


def _consistent(
    header: ReleaseBenchmarkIdentities, members: list[BenchmarkMember]
) -> bool:
    """Tell whether checksums, counts, unique paths and ordering all agree."""
    paths = {member.member_path for member in members}
    return (
        header.identities_checksum == identities_checksum(header)
        and header.members_checksum == members_checksum(members)
        and len(members) == header.member_count
        and len(paths) == len(members)
        and _validation_members_ordered(header, members)
        and _ranked_members_ordered(header, members)
    )


def load_benchmark_identities(
    directory: Path,
) -> tuple[ReleaseBenchmarkIdentities, list[BenchmarkMember]]:
    """Read a frozen folder and prove its checksums, counts and member order."""
    try:
        data = read_bounded(
            contained_file(directory, IDENTITIES_FILE_NAME), MAXIMUM_IDENTITIES_BYTES
        )
        header = ReleaseBenchmarkIdentities.model_validate_json(data)
        if canonical_json(header.model_dump(mode="json")) != data:
            raise ValueError("The identities header is not canonical JSON.")
        members = read_records(directory, MEMBERS_FILE_NAME, BenchmarkMember)
    except (OSError, ValueError, MemoryError, ApplicationFailure):
        raise benchmark_failure("benchmark_invalid") from None
    if not _consistent(header, members):
        raise benchmark_failure("benchmark_invalid")
    return header, members


def validate_benchmark(
    directory: Path, annotations_directory: Path | None = None
) -> BenchmarkValidationReport:
    """Confirm a frozen folder, re-deriving it when annotations are supplied."""
    header, members = load_benchmark_identities(directory)
    integrity: Integrity = "verified"
    if annotations_directory is not None:
        derived, _ = derive_benchmark(
            annotations_directory,
            source_archives=header.source_archives,
            seed=header.selection_rule.seed,
            training_source_count=header.training_source_count,
            reserve_count=header.reserve_count,
            expect_official_counts=False,
        )
        if (
            derived.identities_checksum != header.identities_checksum
            or derived.members_checksum != header.members_checksum
        ):
            raise benchmark_failure("benchmark_mismatch")
        integrity = "verified_with_annotations"
    logger.info("benchmark_identities_verified")
    return BenchmarkValidationReport(
        identities_checksum=header.identities_checksum,
        members_checksum=header.members_checksum,
        validation_count=header.validation_count,
        training_source_count=header.training_source_count,
        reserve_count=header.reserve_count,
        member_count=header.member_count,
        integrity=integrity,
    )
