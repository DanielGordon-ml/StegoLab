"""Safe bounded discovery, immutable source reads, and published CSV splits."""

import hashlib
import os
from pathlib import Path

import pytest

from backend_service.dataset_files import (
    hash_source_file,
    read_source_bytes,
    validate_source_output,
)
from backend_service.dataset_metadata import read_uhd_metadata
from backend_service.dataset_scan import scan_source
from backend_service.failures import ApplicationFailure


def test_scan_counts_sidecars_and_rejections(tmp_path: Path) -> None:
    """All directory entries and source bytes spend the bounded scan allowance."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.PNG").write_bytes(b"pixels")
    (tmp_path / ".DS_Store").write_bytes(b"store")
    (tmp_path / "metadata.csv").write_bytes(b"metadata")
    (tmp_path / "invalid.gif").write_bytes(b"gif")
    inventory = scan_source(tmp_path)
    assert inventory.scanned_entries == 5
    assert inventory.source_bytes == 22
    assert [item.relative_path for item in inventory.files] == [
        ".DS_Store",
        "invalid.gif",
        "metadata.csv",
        "nested/a.PNG",
    ]
    assert [item.kind for item in inventory.files] == [
        "sidecar",
        "unsupported",
        "sidecar",
        "image",
    ]
    assert inventory.files[1].rejection_reason == "unsupported_file_type"
    assert read_source_bytes(tmp_path, inventory.files[3]) == b"pixels"
    with pytest.raises(ApplicationFailure, match="limits"):
        scan_source(tmp_path, maximum_entries=4)
    with pytest.raises(ApplicationFailure, match="limits"):
        scan_source(tmp_path, maximum_source_bytes=21)
    limited = scan_source(tmp_path, maximum_file_bytes=4)
    assert limited.files[3].rejection_reason == "file_size_limit"


@pytest.mark.parametrize("entry", ["symlink", "fifo", "directory_link"])
def test_scan_rejects_unsafe_entries(tmp_path: Path, entry: str) -> None:
    """No source link or special file can be traversed or read."""
    if entry == "fifo":
        os.mkfifo(tmp_path / "file.png")
    elif entry == "directory_link":
        (tmp_path / "dir").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "dir", target_is_directory=True)
    else:
        (tmp_path / "link.png").symlink_to(tmp_path / "missing")
    with pytest.raises(ApplicationFailure):
        scan_source(tmp_path)


def test_read_detects_replacement_and_changes(tmp_path: Path) -> None:
    """A filename cannot silently switch to a new source after inventory."""
    source = tmp_path / "a.png"
    source.write_bytes(b"original")
    file = scan_source(tmp_path).files[0]
    source.unlink()
    source.write_bytes(b"replaced")
    with pytest.raises(ApplicationFailure, match="changed"):
        read_source_bytes(tmp_path, file)
    file = scan_source(tmp_path).files[0]
    source.write_bytes(b"different length")
    with pytest.raises(ApplicationFailure, match="changed"):
        read_source_bytes(tmp_path, file)


def test_read_detects_change_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recheck descriptor and path identities after the exact bytes are read."""
    source = tmp_path / "a.png"
    source.write_bytes(b"original")
    file = scan_source(tmp_path).files[0]
    real_read = os.read
    changed = False

    def mutate(descriptor: int, length: int) -> bytes:
        """Replace source bytes during a bounded read to test detection."""
        nonlocal changed
        data = real_read(descriptor, length)
        if not changed:
            changed = True
            source.write_bytes(b"modified")
        return data

    monkeypatch.setattr(os, "read", mutate)
    with pytest.raises(ApplicationFailure, match="changed"):
        read_source_bytes(tmp_path, file)


def test_hash_rejected_file_streams_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hashing does not need to retain the complete oversized file in memory."""
    data = b"x" * (2 * 1024**2 + 19)
    (tmp_path / "unsupported.bin").write_bytes(data)
    source = scan_source(tmp_path).files[0]
    real_read = os.read
    lengths: list[int] = []

    def track_read(descriptor: int, length: int) -> bytes:
        """Measure read bounds while preserving the real read operation."""
        lengths.append(length)
        return real_read(descriptor, length)

    monkeypatch.setattr(os, "read", track_read)
    assert hash_source_file(tmp_path, source) == hashlib.sha256(data).hexdigest()
    assert len(lengths) >= 3
    assert max(lengths) <= 1024**2
    with pytest.raises(ApplicationFailure, match="limits"):
        hash_source_file(tmp_path, source, limit=10)
    (tmp_path / "unsupported.bin").write_bytes(b"changed")
    with pytest.raises(ApplicationFailure, match="changed"):
        hash_source_file(tmp_path, source)


def test_parent_links_and_overlap_are_rejected(tmp_path: Path) -> None:
    """Reject overlap in either direction and symlinks in parent components."""
    source = tmp_path / "source"
    source.mkdir()
    for output in (source, source / "output", tmp_path):
        with pytest.raises(ApplicationFailure):
            validate_source_output(source, output)
    (tmp_path / "alias").symlink_to(source, target_is_directory=True)
    with pytest.raises(ApplicationFailure):
        scan_source(tmp_path / "alias")
    validate_source_output(source, tmp_path / "output")


def test_uhd_metadata_preserves_blank_subsets_and_exact_splits(tmp_path: Path) -> None:
    """Most published rows have blank subsets; these remain valid source data."""
    data = (
        b"image_name,set,subset,quality_mos\n"
        b"a.JPG,training,,0.2\n"
        b"b.JPG,validation,validation,0.3\n"
        b"c.JPG,test,test,\n"
    )
    (tmp_path / "uhd-iqa-metadata.csv").write_bytes(data)
    result = read_uhd_metadata(tmp_path, "uhd-iqa-metadata.csv", scan_source(tmp_path))
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    assert result.rows["a.JPG"].subset == ""
    assert result.rows["a.JPG"].upstream_identity == "uhd_iqa:a.JPG"
    assert [row.assigned_split for row in result.rows.values()] == [
        "train",
        "tuning",
        "held_out",
    ]


@pytest.mark.parametrize(
    "data",
    [
        "image_name,set\na.jpg,training\n",
        "image_name,set,subset\na.jpg,unknown,\n",
        "image_name,set,subset\na.jpg,train,\n",
        "image_name,set,subset\na.jpg,training,\na.jpg,test,\n",
        "image_name,set,subset\n../a.jpg,training,\n",
        "image_name,set,subset\n/a.jpg,training,\n",
        "image_name,set,subset\na.jpg,training,,extra\n",
        "image_name,set,set,subset\na.jpg,training,training,\n",
    ],
)
def test_uhd_metadata_rejects_ambiguous_records(tmp_path: Path, data: str) -> None:
    """Bad identities, missing keys, duplicates, and unknown splits fail clearly."""
    (tmp_path / "metadata.csv").write_text(data)
    with pytest.raises(ApplicationFailure):
        read_uhd_metadata(tmp_path, "metadata.csv", scan_source(tmp_path))
