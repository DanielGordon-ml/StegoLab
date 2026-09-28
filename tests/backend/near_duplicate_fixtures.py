"""Seeded smooth images, tiny prepared revisions and frozen identity folders."""

import hashlib
from collections.abc import Sequence
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from backend_service.benchmark_freeze import (
    IDENTITIES_FILE_NAME,
    MEMBERS_FILE_NAME,
    POLICY_VERSION,
    identities_checksum,
    members_checksum,
    selection_key,
)
from backend_service.dataset_image import prepare_dataset_image
from backend_service.dataset_manifest import write_manifest
from backend_service.dataset_serialization import canonical_json
from schemas.benchmark_identities import BenchmarkMember, ReleaseBenchmarkIdentities
from schemas.datasets import DatasetImageRecord, DatasetPreparationRequest
from schemas.near_duplicate_audit import NearDuplicateAuditRequest

RevisionEntry = tuple[str, Image.Image, str | None]
REMOTE_REQUEST = {
    "source_kind": "hugging_face",
    "source_url": "https://example.invalid/fixture",
    "terms_reference": "fixture_terms",
    "training_intended": False,
}
LOCAL_REQUEST = {"source_kind": "local", "training_intended": False}
ARCHIVE = {"repository": "fixture/coco-mirror", "revision": "a" * 40}
ARCHIVE |= {"archive_path": "val2017.zip", "archive_sha256": "b" * 64}
ANNOTATION = {"path": "annotations/captions_val2017.json", "sha256": "c" * 64}


def smooth_image(seed: int, size: int = 320) -> Image.Image:
    """Enlarge seeded low-resolution noise bilinearly into a smooth RGB picture."""
    coarse = np.random.default_rng(seed).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    with Image.fromarray(coarse, "RGB") as small:
        return small.resize((size, size), Image.Resampling.BILINEAR)


def resized_variant(image: Image.Image, size: int = 280) -> Image.Image:
    """Return the same picture at another resolution."""
    return image.resize((size, size), Image.Resampling.BILINEAR)


def recompressed_variant(image: Image.Image, quality: int = 60) -> Image.Image:
    """Return the picture after one lossy JPEG round trip."""
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=quality)
    with Image.open(BytesIO(stream.getvalue())) as reopened:
        return reopened.convert("RGB")


def shifted_crop_variant(image: Image.Image, offset: int = 24) -> Image.Image:
    """Return a crop shifted by a few pixels and enlarged back to the full size."""
    with image.crop((offset, offset, *image.size)) as cropped:
        return cropped.resize(image.size, Image.Resampling.BILINEAR)


def leaky_entries() -> list[RevisionEntry]:
    """Describe a revision with a within-split near duplicate and a cross-split leak."""
    return [
        ("training/000.png", smooth_image(0), "training"),
        ("training/001.png", smooth_image(1), "training"),
        ("training/002.png", recompressed_variant(smooth_image(1)), "training"),
        ("validation/003.png", smooth_image(2), "validation"),
        ("test/004.png", resized_variant(smooth_image(0)), "test"),
        ("validation/005.png", smooth_image(4), "validation"),
    ]


def audit_request(
    revision: Path, output_root: Path | None, **extra: str | int
) -> NearDuplicateAuditRequest:
    """Build a request that writes its report under the given root, if any."""
    fields: dict[str, str | int] = {"dataset_directory": str(revision)}
    if output_root is not None:
        fields["output_root"] = str(output_root)
    return NearDuplicateAuditRequest.model_validate(fields | extra)


def snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """List every entry of a revision with its size and modification time."""
    return {
        str(path.relative_to(directory)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(directory.rglob("*"))
    }


def _record(
    directory: Path, index: int, entry: RevisionEntry, prefix: str | None
) -> DatasetImageRecord:
    """Prepare one entry into the revision and describe it like an import would."""
    source_path, image, split = entry
    stream = BytesIO()
    image.save(stream, format="PNG")
    content, relative = stream.getvalue(), f"images/{index:04d}.png"
    prepared = prepare_dataset_image(content, directory / relative, lambda size: None)
    return DatasetImageRecord(
        source_path=source_path,
        prepared_path=relative,
        source_identity=None if prefix is None else f"{prefix}:{source_path}",
        upstream_split=split,
        source_checksum=hashlib.sha256(content).hexdigest(),
        source_bytes=len(content),
        **asdict(prepared),
    )


def write_revision(
    directory: Path,
    entries: Sequence[RevisionEntry],
    *,
    dataset_name: str = "audit_fixture",
    identity_prefix: str | None = "fixture",
) -> Path:
    """Prepare (source path, image, split or None) entries into a named revision.

    Splits give a UHD-IQA style revision; no splits give a fetched-archive style
    revision with generated splits. Identities are ``<prefix>:<source path>``;
    without a prefix the records are local ones that carry no identity.
    """
    (directory / "images").mkdir(parents=True)
    records = [
        _record(directory, index, entry, identity_prefix)
        for index, entry in enumerate(entries)
    ]
    declared = any(split is not None for _, _, split in entries)
    source = REMOTE_REQUEST if identity_prefix is not None else LOCAL_REQUEST
    request = DatasetPreparationRequest.model_validate(
        {"source_directory": "unused", "dataset_name": dataset_name}
        | ({} if declared else source)
    )
    manifest = write_manifest(
        directory,
        request,
        records,
        [],
        before_write=lambda size: None,
        metadata_checksum="a" * 64 if declared else None,
        expected_images=len(records) if declared else None,
    )
    directory.rename(directory.parent / manifest.revision)
    return directory.parent / manifest.revision


def _member(path: str, role: str, rank: int | None) -> BenchmarkMember:
    """Describe one COCO member whose identifier and split are spelled in its path."""
    fields = {"declared_width": 640, "declared_height": 480, "role": role, "rank": rank}
    identity = {"coco_image_identifier": int(path[-16:-4]), "source_split": path[:-17]}
    return BenchmarkMember.model_validate(fields | identity | {"member_path": path})


def write_identities(
    directory: Path,
    validation: Sequence[int],
    population: Sequence[int],
    *,
    training_source_count: int,
    reserve_count: int,
    seed: int = 0,
) -> ReleaseBenchmarkIdentities:
    """Write a tiny frozen folder with the checksums the freeze module writes."""
    ranked = sorted(
        (f"train2017/{number:012d}.jpg" for number in population),
        key=lambda path: (selection_key(POLICY_VERSION, seed, path), path),
    )
    members = [
        _member(f"val2017/{number:012d}.jpg", "validation", None)
        for number in sorted(validation)
    ]
    for rank, path in enumerate(ranked[: training_source_count + reserve_count]):
        role = "training_source" if rank < training_source_count else "reserve"
        members.append(_member(path, role, rank))
    counts = {
        "training_source_count": training_source_count,
        "reserve_count": reserve_count,
    }
    header = ReleaseBenchmarkIdentities.model_validate(
        {
            "source_archives": [ARCHIVE | {"archive_bytes": 1}],
            "annotation_files": [
                ANNOTATION | {"byte_count": 1, "image_count": len(validation)}
            ],
            "selection_rule": counts
            | {"seed": seed, "population_count": len(population)},
            "validation_count": len(validation),
            "member_count": len(members),
            "members_checksum": members_checksum(members),
            "identities_checksum": "0" * 64,
        }
        | counts
    )
    header.identities_checksum = identities_checksum(header)
    directory.mkdir(parents=True)
    (directory / IDENTITIES_FILE_NAME).write_bytes(
        canonical_json(header.model_dump(mode="json"))
    )
    (directory / MEMBERS_FILE_NAME).write_bytes(
        b"".join(canonical_json(item.model_dump(mode="json")) for item in members)
    )
    return header
