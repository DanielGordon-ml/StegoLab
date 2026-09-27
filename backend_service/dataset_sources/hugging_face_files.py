"""Understand Hugging Face Hub listings and choose the files a fetch downloads."""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

from backend_service.failures import ApplicationFailure
from schemas.dataset_common import DatasetSplit, relative_path
from schemas.dataset_sources import HuggingFaceSourceSpec

HUB_HOST = "huggingface.co"
ALLOWED_EXTENSIONS = frozenset({"parquet", "png", "jpg", "jpeg", "txt", "zip"})
SHARD_NAME = re.compile(r"^(?P<split>train|validation|test)-\d{5}-of-\d{5}\.parquet$")
LFS_OBJECT_IDENTIFIER = re.compile(r"^[a-f0-9]{64}$")
REVISION_SHA = re.compile(r"^[a-f0-9]{40}$")
NEXT_LINK = re.compile(r'<([^>]*)>\s*;[^,]*rel="?next"?', re.IGNORECASE)
DEFAULT_SPLIT_ASSIGNMENT: dict[str, DatasetSplit] = {
    "train": "train",
    "validation": "tuning",
    "test": "held_out",
}
_MESSAGES = {
    "source_unavailable": (
        "The Hugging Face Hub did not answer as expected. Try again in a moment; "
        "if the problem continues, check the repository name and revision."
    ),
    "source_listing_invalid": (
        "The Hugging Face Hub returned a listing that is missing, too large or "
        "not in the expected form. Try again later."
    ),
    "source_listing_limit": (
        "The repository has more files than can be listed. Narrow the selection "
        "with a path prefix or explicit file names."
    ),
    "source_files_missing": (
        "Some of the requested files are not in the repository at this revision. "
        "Check the file names and the revision."
    ),
    "source_files_unsupported": (
        "Some of the requested files are not parquet, png, jpg, jpeg, txt or zip "
        "files. Name supported files only."
    ),
    "source_files_empty": (
        "No supported files matched the request at this revision. Check the path "
        "prefix, split and file names."
    ),
}
_STATUS_CODES = {
    "source_unavailable": 502,
    "source_listing_invalid": 502,
    "source_files_missing": 404,
}


def hugging_face_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for a Hugging Face source failure code."""
    return ApplicationFailure(code, _MESSAGES[code], _STATUS_CODES.get(code, 422))


@dataclass(frozen=True)
class TreeEntry:
    """One file of the repository tree with what the Hub knows about it."""

    path: str
    size: int | None
    sha256: str | None


@dataclass(frozen=True)
class ParsedTreePage:
    """Files read from the listing and how many unsafe names were skipped."""

    entries: tuple[TreeEntry, ...]
    skipped_unsafe: int


@dataclass(frozen=True)
class FileSelection:
    """The files kept for a fetch, in a fixed order, with any selection notes."""

    entries: tuple[TreeEntry, ...]
    warnings: tuple[str, ...]


def decode_document(body: bytes) -> object:
    """Parse a JSON body, reporting damaged or non-JSON answers plainly."""
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise hugging_face_failure("source_listing_invalid") from None


def revision_sha(document: object) -> str:
    """Read the 40-character commit sha from the revision document."""
    sha = document.get("sha") if isinstance(document, dict) else None
    if not isinstance(sha, str) or REVISION_SHA.fullmatch(sha) is None:
        raise hugging_face_failure("source_listing_invalid")
    return sha


def next_page_url(link_header: str | None) -> str | None:
    """Return the next listing page named by a Link header, if there is one."""
    match = None if link_header is None else NEXT_LINK.search(link_header)
    if match is None:
        return None
    url = match.group(1).strip()
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != HUB_HOST:
        raise hugging_face_failure("source_listing_invalid")
    return url


def parse_tree_page(document: object) -> ParsedTreePage:
    """Turn one page of the tree listing into file entries, skipping folders."""
    if not isinstance(document, list):
        raise hugging_face_failure("source_listing_invalid")
    entries: list[TreeEntry] = []
    skipped = 0
    for item in document:
        if not isinstance(item, dict):
            raise hugging_face_failure("source_listing_invalid")
        if item.get("type") != "file":
            continue
        path = item.get("path")
        if not isinstance(path, str):
            raise hugging_face_failure("source_listing_invalid")
        try:
            relative_path(path)
        except ValueError:
            skipped += 1
            continue
        entries.append(TreeEntry(path, _entry_size(item), _entry_sha256(item)))
    return ParsedTreePage(tuple(entries), skipped)


def _entry_size(item: dict[str, object]) -> int | None:
    """Prefer the large-file size, then the plain size, when either is sound."""
    lfs = item.get("lfs")
    candidates = [lfs.get("size") if isinstance(lfs, dict) else None, item.get("size")]
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _entry_sha256(item: dict[str, object]) -> str | None:
    """Return the large-file object identifier when it is a sha256 digest."""
    lfs = item.get("lfs")
    if not isinstance(lfs, dict):
        return None
    identifier = lfs.get("oid")
    if isinstance(identifier, str) and LFS_OBJECT_IDENTIFIER.fullmatch(identifier):
        return identifier
    return None


def extension_allowed(path: str) -> bool:
    """Report whether the file type is one the materializer can use."""
    basename = path.rsplit("/", 1)[-1]
    if "." not in basename:
        return False
    return basename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS


def shard_split(path: str) -> str | None:
    """Return the Hub split named by a parquet shard file, if the name fits."""
    match = SHARD_NAME.fullmatch(path.rsplit("/", 1)[-1])
    return None if match is None else match.group("split")


def _matches_filters(path: str, spec: HuggingFaceSourceSpec) -> bool:
    """Apply the prefix, allowed types and the optional split shard pattern."""
    prefix = spec.path_prefix
    inside = prefix is None or path == prefix or path.startswith(prefix + "/")
    if not inside or not extension_allowed(path):
        return False
    return spec.split is None or shard_split(path) == spec.split


def select_files(
    entries: Iterable[TreeEntry], spec: HuggingFaceSourceSpec
) -> FileSelection:
    """Pick the files the request asks for, sorted by path and capped in count.

    Explicit file names win over the prefix and split filters and must all
    exist with a supported type. Otherwise every supported file under the
    prefix is kept, narrowed to the named split's parquet shards when a split
    is requested. More matches than ``maximum_files`` keep the first ones in
    path order and add a warning.
    """
    by_path = {entry.path: entry for entry in entries}
    if spec.file_names is not None:
        if any(name not in by_path for name in spec.file_names):
            raise hugging_face_failure("source_files_missing")
        if any(not extension_allowed(name) for name in spec.file_names):
            raise hugging_face_failure("source_files_unsupported")
        chosen = [by_path[name] for name in sorted(spec.file_names)]
    else:
        chosen = [
            by_path[path] for path in sorted(by_path) if _matches_filters(path, spec)
        ]
    if not chosen:
        raise hugging_face_failure("source_files_empty")
    warnings: tuple[str, ...] = ()
    if len(chosen) > spec.maximum_files:
        warnings = (
            f"Only the first {spec.maximum_files} of {len(chosen)} matching files "
            "are selected. Raise maximum_files or narrow the selection.",
        )
        chosen = chosen[: spec.maximum_files]
    return FileSelection(tuple(chosen), warnings)


def split_assignments(
    spec: HuggingFaceSourceSpec, entries: Iterable[TreeEntry]
) -> tuple[dict[str, DatasetSplit], dict[str, str]]:
    """Map shard splits to assigned dataset splits when a split was requested.

    The first mapping goes from the Hub split label to the dataset split,
    honouring the request's ``archive_splits`` when it names the label. The
    second maps each shard basename to its Hub split label.
    """
    if spec.split is None:
        return {}, {}
    requested = spec.archive_splits or {}
    split_mapping: dict[str, DatasetSplit] = {}
    member_split_labels: dict[str, str] = {}
    for entry in entries:
        label = shard_split(entry.path)
        if label is None:
            continue
        split_mapping[label] = requested.get(label, DEFAULT_SPLIT_ASSIGNMENT[label])
        member_split_labels[entry.path.rsplit("/", 1)[-1]] = label
    return split_mapping, member_split_labels
