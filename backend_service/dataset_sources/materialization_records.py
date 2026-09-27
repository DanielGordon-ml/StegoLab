"""Bookkeeping for one materialization: kept files, rejections and metadata."""

import codecs
import csv
import io
import os
import stat
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

from backend_service.dataset_sources.archive_members import (
    JPEG_MAGIC,
    PNG_MAGIC,
    SNIFF_BYTES,
    MemberPolicy,
    classify_member,
    looks_like_archive,
)
from backend_service.dataset_sources.cache import CacheEntry
from backend_service.dataset_sources.plans import (
    MaterializationPlan,
    PlannedAsset,
    asset_identity,
)
from backend_service.failures import ApplicationFailure
from schemas.dataset_common import DatasetSplit
from schemas.dataset_fetch import TextCorpusSummary
from schemas.dataset_sources import PlannedAssetRecord, SourceMarker

AssetKind = Literal["archive", "parquet", "image", "text", "rejected"]
Assets = list[tuple[PlannedAsset, CacheEntry]]
PARQUET_MAGIC = b"PAR1"
HUB_DATASET_PREFIX = "https://huggingface.co/datasets/"
IMAGE_LIMIT_REASON = "image_limit"
MAXIMUM_SUBSET_LENGTH = 255
READ_CHUNK_BYTES = 1024**2
_MESSAGES = {
    "source_folder_occupied": (
        "The folder for this source name already holds files from a different "
        "request or source. Choose another source name."
    ),
    "source_asset_missing": (
        "A file this fetch needs is missing from the download cache. Retry the fetch."
    ),
    "source_member_limit": (
        "The source has more files than one fetch may handle. Select a smaller subset."
    ),
    "source_unreadable": (
        "A fetched file could not be read back. Clear the download cache and retry."
    ),
}
_STATUS_CODES = {"source_folder_occupied": 409, "source_member_limit": 400}


def materialization_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for a materialization failure code."""
    return ApplicationFailure(code, _MESSAGES[code], _STATUS_CODES.get(code, 502))


@dataclass(frozen=True)
class MaterializedSource:
    """A published raw folder, its marker, and what the run kept or skipped."""

    directory: Path
    marker: SourceMarker
    reused: bool
    text_corpus: TextCorpusSummary | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class KeptImage:
    """One image saved in the raw folder and what the metadata says about it."""

    path: str
    label: str | None
    subset: str
    identity: str


@dataclass
class MaterializationLedger:
    """Count what one materialization kept, skipped and wrote."""

    maximum_images: int
    maximum_members: int
    images: list[KeptImage] = field(default_factory=list)
    image_paths: set[str] = field(default_factory=set)
    rejections: Counter[str] = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)
    text_files: int = 0
    text_characters: int = 0
    text_bytes: int = 0
    bytes_written: int = 0

    @property
    def member_count(self) -> int:
        """Return how many files were kept."""
        return len(self.images) + self.text_files

    @property
    def rejected_count(self) -> int:
        """Return how many members or rows were skipped."""
        return sum(self.rejections.values())

    def _check_members(self) -> None:
        """Fail plainly when the source holds more members than one fetch may."""
        if self.member_count + self.rejected_count > self.maximum_members:
            raise materialization_failure("source_member_limit")

    def reject(self, reason: str) -> None:
        """Count one skipped member under its reason."""
        self.rejections[reason] += 1
        self._check_members()

    def image_allowed(self) -> bool:
        """Tell whether another image may be kept under the image cap."""
        return len(self.images) < self.maximum_images

    def keep_image(self, image: KeptImage, size: int) -> None:
        """Record one saved image and the bytes it took."""
        self.images.append(image)
        self.image_paths.add(image.path)
        self.bytes_written += size
        self._check_members()

    def keep_text(self, characters: int, size: int) -> None:
        """Record one saved text file with its character and byte counts."""
        self.text_files += 1
        self.text_characters += characters
        self.text_bytes += size
        self.bytes_written += size
        self._check_members()

    def text_summary(self) -> TextCorpusSummary | None:
        """Describe the saved text files, or None when there are none."""
        if self.text_files == 0:
            return None
        files, characters, size = self.text_files, self.text_characters, self.text_bytes
        return TextCorpusSummary(files=files, characters=characters, bytes=size)

    def rejection_reasons(self) -> dict[str, int]:
        """Return rejection counts by reason in a stable order."""
        return dict(sorted(self.rejections.items()))


def metadata_csv(
    images: list[KeptImage], split_mapping: dict[str, DatasetSplit]
) -> bytes | None:
    """Build the split CSV only when every image has a mapped label and unique name."""
    by_name = {image.path.rsplit("/", 1)[-1]: image for image in images}
    unmapped = any(image.label not in split_mapping for image in images)
    if not images or len(by_name) != len(images) or unmapped:
        return None
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["image_name", "set", "subset", "identity"])
    for name in sorted(by_name):
        image = by_name[name]
        writer.writerow([name, image.label, image.subset, image.identity])
    return buffer.getvalue().encode("utf-8")


def build_marker(
    plan: MaterializationPlan,
    assets: Assets,
    ledger: MaterializationLedger,
    *,
    declared_splits: bool,
    created_at: str,
) -> SourceMarker:
    """Describe the published folder for later reuse and manifest provenance."""
    resolved = plan.resolved
    return SourceMarker(
        source_kind=resolved.source_kind,
        source_name=plan.source_name,
        reference=resolved.reference,
        source_url=resolved.reference,
        terms_reference=resolved.terms_reference,
        resolved_revision=resolved.resolved_revision,
        materialization_identity=plan.identity,
        content=resolved.content,
        assets=[
            PlannedAssetRecord(
                path=asset.path,
                size_bytes=entry.manifest.size_bytes,
                sha256=entry.manifest.sha256,
            )
            for asset, entry in assets
        ],
        member_count=ledger.member_count,
        rejected_member_count=ledger.rejected_count,
        split_mapping=dict(resolved.split_mapping) if declared_splits else {},
        declared_splits=declared_splits,
        created_at=created_at,
        completed=True,
    )


def entry_for(
    plan: MaterializationPlan, asset: PlannedAsset, entries: dict[str, CacheEntry]
) -> CacheEntry:
    """Find the cached file of one asset by its identity, or else by its path."""
    resolved = plan.resolved
    pinned = (resolved.source_kind, resolved.reference, resolved.resolved_revision)
    identity = asset_identity(*pinned, asset.path, asset.expected_sha256)
    entry = entries.get(identity, entries.get(asset.path))
    if entry is None:
        raise materialization_failure("source_asset_missing")
    return entry


def open_regular_file(path: Path) -> IO[bytes]:
    """Open a file for reading without following links; refuse special files."""
    try:
        stream = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb")
    except OSError:
        raise materialization_failure("source_unreadable") from None
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        stream.close()
        raise materialization_failure("source_unreadable")
    return stream


def stream_chunks(stream: IO[bytes]) -> Iterator[bytes]:
    """Yield a readable stream in bounded chunks."""
    while chunk := stream.read(READ_CHUNK_BYTES):
        yield chunk


def count_characters(path: Path) -> int:
    """Count the characters of a saved text file one bounded chunk at a time."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    total = 0
    try:
        with open_regular_file(path) as stream:
            for chunk in stream_chunks(stream):
                total += len(decoder.decode(chunk))
    except OSError:
        raise materialization_failure("source_unreadable") from None
    return total + len(decoder.decode(b"", final=True))


def classify_asset(
    asset: PlannedAsset, entry: CacheEntry, policy: MemberPolicy
) -> tuple[AssetKind, str | None]:
    """Tell whether an asset is an archive, a parquet file, a loose file or junk."""
    try:
        with open_regular_file(entry.file) as stream:
            head = stream.read(SNIFF_BYTES)
    except OSError:
        raise materialization_failure("source_unreadable") from None
    if looks_like_archive(head):
        return "archive", None
    if head.startswith(PARQUET_MAGIC):
        return "parquet", None
    decision = classify_member(asset.path, head, policy)
    return decision.kind, decision.reason


def image_extension(data: bytes) -> str | None:
    """Name the image format of raw bytes by their leading bytes, or None."""
    if data.startswith(PNG_MAGIC):
        return "png"
    return "jpg" if data.startswith(JPEG_MAGIC) else None


def member_key(member_path: str, asset_basename: str) -> str:
    """Return the folder or archive name that carries a member's split label."""
    parts = member_path.split("/")
    return parts[0] if len(parts) > 1 else asset_basename


def asset_label(plan: MaterializationPlan, asset: PlannedAsset) -> str | None:
    """Look up the split label of a whole asset, such as a parquet shard."""
    labels = plan.resolved.member_split_labels
    folder = member_key(asset.path, asset.basename)
    return labels.get(asset.basename, labels.get(asset.path, labels.get(folder)))


def row_identity(plan: MaterializationPlan, asset: PlannedAsset, row: int) -> str:
    """Name one parquet row by repository, revision, shard path and row number."""
    resolved = plan.resolved
    repository = plan.source_name
    if resolved.reference.startswith(HUB_DATASET_PREFIX):
        remainder = resolved.reference[len(HUB_DATASET_PREFIX) :]
        repository = remainder.split("/tree/", 1)[0]
    return f"{repository}@{resolved.resolved_revision}:{asset.path}#{row}"
