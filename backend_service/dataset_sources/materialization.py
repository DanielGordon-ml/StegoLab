"""Turn cached downloads into one published raw source folder with provenance."""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from backend_service.dataset_files import dataset_failure, relative_parts
from backend_service.dataset_serialization import checksum
from backend_service.dataset_sources import materialization_records as records
from backend_service.dataset_sources.archive_extraction import extract_archive
from backend_service.dataset_sources.archive_members import (
    MemberPolicy,
    archive_failure,
)
from backend_service.dataset_sources.cache import CacheEntry
from backend_service.dataset_sources.parquet_reader import (
    inspect_parquet_columns,
    iter_image_rows,
    iter_text_rows,
    parquet_failure,
)
from backend_service.dataset_sources.plans import MaterializationPlan, PlannedAsset
from backend_service.dataset_sources.source_marker import (
    METADATA_NAME,
    raw_folder_state,
    read_source_marker,
    write_source_marker,
)
from backend_service.dataset_sources.transfer import DownloadStopped
from backend_service.dataset_storage import DatasetWriter
from backend_service.failures import ApplicationFailure

logger = logging.getLogger(__name__)

MaterializedSource = records.MaterializedSource
DEFAULT_MEMBER_POLICY = MemberPolicy()
EXTRACTION_GROWTH_FACTOR = 8
MINIMUM_EXTRACTION_BYTES = 256 * 1024**2
ProgressCallback = Callable[[int, int | None], None]
SPLIT_LABELS_WARNING = (
    "Split labels were not saved because some images have no label or share a "
    "file name; splits will be generated when the dataset is prepared."
)


def find_materialized(
    data_root: Path, plan: MaterializationPlan
) -> MaterializedSource | None:
    """Reuse the folder whose completed marker has this identity; refuse others."""
    state = raw_folder_state(data_root, plan.source_name, plan.identity)
    if state == "available":
        return None
    directory = data_root / plan.source_name
    marker = read_source_marker(directory) if state == "reusable" else None
    if marker is None:
        raise records.materialization_failure("source_folder_occupied")
    logger.info("source_folder_reused")
    return MaterializedSource(directory, marker, True)


@dataclass
class _Run:
    """Everything one materialization run shares between its assets."""

    plan: MaterializationPlan
    writer: DatasetWriter
    ledger: records.MaterializationLedger
    policy: MemberPolicy
    progress: ProgressCallback | None
    stop: Callable[[], bool]

    def check_stop(self) -> None:
        """Raise ``DownloadStopped`` between members when the caller asked."""
        if self.stop():
            logger.info("materialization_stopped")
            raise DownloadStopped()

    def report(self, done: int | None = None, total: int | None = None) -> None:
        """Tell the caller how many bytes now sit in the stage."""
        if self.progress is not None:
            self.progress(self.ledger.bytes_written if done is None else done, total)

    def staged(self, path: str) -> Path:
        """Locate one written member inside the stage."""
        return self.writer.stage.joinpath(*relative_parts(path))


def _keep(
    run: _Run, path: str, kind: str, size: int, label: str | None, key: str
) -> None:
    """Account for one file in the stage, dropping images past the image cap."""
    if kind == "text":
        run.ledger.keep_text(records.count_characters(run.staged(path)), size)
    elif run.ledger.image_allowed():
        subset = key[: records.MAXIMUM_SUBSET_LENGTH]
        identity = f"{run.plan.source_name}:{path}"
        run.ledger.keep_image(records.KeptImage(path, label, subset, identity), size)
    else:
        try:
            run.staged(path).unlink()
        except OSError:
            raise dataset_failure("dataset_storage") from None
        run.ledger.reject(records.IMAGE_LIMIT_REASON)


def extraction_bound(policy: MemberPolicy, archive_bytes: int) -> int:
    """Cap the bytes one archive may extract relative to its own size.

    Images hardly compress, so a fetched archive that grows many times over
    is a bomb rather than a dataset; the floor keeps tiny archives usable.
    """
    relative = max(MINIMUM_EXTRACTION_BYTES, EXTRACTION_GROWTH_FACTOR * archive_bytes)
    return min(policy.maximum_total_bytes, relative)


def _extract(run: _Run, asset: PlannedAsset, entry: CacheEntry) -> None:
    """Stream an archive's members into the stage and account for each one."""
    labels, base = run.plan.resolved.member_split_labels, run.ledger.bytes_written

    def split_for_member(name: str) -> str | None:
        """Label a member by its first folder, or by the archive name when flat."""
        return labels.get(records.member_key(name, asset.basename))

    def report(done: int, declared: int | None) -> None:
        """Offset the archive's own progress by the bytes written before it."""
        run.report(base + done, None if declared is None else base + declared)

    bound = extraction_bound(run.policy, entry.manifest.size_bytes)
    extraction = extract_archive(
        entry.file,
        run.writer,
        policy=replace(run.policy, maximum_total_bytes=bound),
        split_for_member=split_for_member,
        progress=report,
        stop=run.stop,
    )
    for rejection in extraction.rejected:
        run.ledger.reject(rejection.reason)
    for member in sorted(extraction.members, key=lambda item: item.path):
        key = records.member_key(member.path, asset.basename)
        _keep(run, member.path, member.kind, member.size, member.split_label, key)


def _copy_loose(run: _Run, asset: PlannedAsset, entry: CacheEntry, kind: str) -> None:
    """Save a loose image or text file under its own path."""
    size = entry.manifest.size_bytes
    if size > run.policy.maximum_bytes_for(kind):
        return run.ledger.reject("member_size_limit")
    with records.open_regular_file(entry.file) as stream:
        chunks = records.stream_chunks(stream)
        run.writer.write_stream(asset.path, chunks, maximum_bytes=size)
    label, key = records.asset_label(run.plan, asset), asset.basename
    _keep(run, asset.path, kind, size, label, records.member_key(asset.path, key))
    run.report()


def _parquet_images(run: _Run, asset: PlannedAsset, entry: CacheEntry) -> None:
    """Save every usable image row as ``<label>/<sha256>.<png|jpg>``."""
    columns = inspect_parquet_columns(
        entry.file, image_column=run.plan.resolved.image_column, text_column=None
    )
    if columns.image_column is None:
        raise parquet_failure("source_schema")
    label = records.asset_label(run.plan, asset)
    subset = asset.basename[: records.MAXIMUM_SUBSET_LENGTH]
    rows = iter_image_rows(
        entry.file,
        columns.image_column,
        maximum_rows=run.policy.maximum_members + 1,
        maximum_bytes=run.policy.maximum_image_bytes,
    )
    for image in rows:
        run.check_stop()
        extension = records.image_extension(image.data)
        if extension is None:
            run.ledger.reject("unsupported_file_type")
        elif not run.ledger.image_allowed():
            run.ledger.reject(records.IMAGE_LIMIT_REASON)
        elif (
            path := f"{label or 'images'}/{checksum(image.data)}.{extension}"
        ) in run.ledger.image_paths:
            run.ledger.reject("duplicate_member")
        else:
            run.writer.write_bytes(path, image.data)
            identity = records.row_identity(run.plan, asset, image.row)
            image_record = records.KeptImage(path, label, subset, identity)
            run.ledger.keep_image(image_record, len(image.data))
            run.report()


def _text_rows(run: _Run, shards: records.Assets) -> Iterator[bytes]:
    """Yield the UTF-8 text rows of the shards joined by single newlines."""
    separator = ""
    for _asset, entry in shards:
        columns = inspect_parquet_columns(
            entry.file, image_column=None, text_column=run.plan.resolved.text_column
        )
        if columns.text_column is None:
            raise parquet_failure("source_schema")
        limit = run.policy.maximum_text_bytes
        rows = iter_text_rows(entry.file, columns.text_column, maximum_bytes=limit)
        for text in rows:
            run.check_stop()
            yield (separator + text).encode("utf-8")
            separator = "\n"


def _text_corpus(run: _Run, shards: records.Assets) -> None:
    """Join the text rows of the shards of each label into one ``.txt`` file."""
    groups: dict[str, records.Assets] = {}
    for asset, entry in shards:
        prefix = asset.path.rsplit("/", 1)[0] if "/" in asset.path else "text"
        label = records.asset_label(run.plan, asset) or "text"
        groups.setdefault(f"{prefix}/{label}.txt", []).append((asset, entry))
    for path, members in sorted(groups.items()):
        rows, limit = _text_rows(run, members), run.policy.maximum_text_bytes
        _digest, size = run.writer.write_stream(path, rows, maximum_bytes=limit)
        run.ledger.keep_text(records.count_characters(run.staged(path)), size)
        run.report()


def _write_metadata(run: _Run) -> bool:
    """Write the split CSV when it can be trusted; say whether splits are declared."""
    data = records.metadata_csv(run.ledger.images, run.plan.resolved.split_mapping)
    if data is not None:
        run.writer.write_bytes(METADATA_NAME, data)
        return True
    if run.ledger.images and run.plan.resolved.declared_splits:
        run.ledger.warnings.append(SPLIT_LABELS_WARNING)
        logger.warning("source_split_labels_not_saved")
    return False


def _materialize_asset(run: _Run, asset: PlannedAsset, entry: CacheEntry) -> bool:
    """Handle one asset by its content; return True when it is a text shard."""
    kind, reason = records.classify_asset(asset, entry, run.policy)
    if run.plan.resolved.source_kind != "hugging_face" and kind != "archive":
        raise archive_failure("archive_unreadable")
    if kind == "archive":
        _extract(run, asset, entry)
    elif kind == "parquet" and run.plan.resolved.content == "text":
        return True
    elif kind == "parquet":
        _parquet_images(run, asset, entry)
    elif kind == "rejected":
        run.ledger.reject(reason or "unsupported_file_type")
    else:
        _copy_loose(run, asset, entry, kind)
    return False


def materialize(
    plan: MaterializationPlan,
    entries: dict[str, CacheEntry],
    *,
    data_root: Path,
    cache_root: Path,
    created_at: str,
    progress: ProgressCallback | None = None,
    stop: Callable[[], bool] = lambda: False,
    policy: MemberPolicy = DEFAULT_MEMBER_POLICY,
) -> MaterializedSource:
    """Publish the raw folder for a plan from cached files, or reuse it.

    ``entries`` maps asset identities (``plans.asset_identity``) to complete
    cache entries. A stop between members raises ``DownloadStopped`` after the
    stage was removed, so nothing is published.
    """
    if (existing := find_materialized(data_root, plan)) is not None:
        return existing
    assets: records.Assets = [
        (asset, records.entry_for(plan, asset, entries))
        for asset in plan.resolved.assets
    ]
    ledger = records.MaterializationLedger(plan.maximum_images, policy.maximum_members)
    with DatasetWriter(data_root, cache_root) as writer:
        run = _Run(plan, writer, ledger, policy, progress, stop)
        text_shards: records.Assets = []
        for asset, entry in assets:
            run.check_stop()
            if _materialize_asset(run, asset, entry):
                text_shards.append((asset, entry))
        if text_shards:
            _text_corpus(run, text_shards)
        declared = _write_metadata(run)
        marker = records.build_marker(
            plan, assets, ledger, declared_splits=declared, created_at=created_at
        )
        write_source_marker(writer, marker)
        try:
            directory = writer.publish(plan.source_name)
        except ApplicationFailure as failure:
            if failure.code != "dataset_exists":
                raise
            raise records.materialization_failure("source_folder_occupied") from None
    logger.info("source_materialized")
    extras = (ledger.text_summary(), ledger.rejection_reasons(), tuple(ledger.warnings))
    return MaterializedSource(directory, marker, False, *extras)
