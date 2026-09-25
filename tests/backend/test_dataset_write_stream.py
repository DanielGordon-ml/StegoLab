"""Streamed stage writes and the shared disk reserve rule."""

import hashlib
import os
import shutil
import stat
from collections import namedtuple
from pathlib import Path

import pytest

from backend_service import disk_reserve
from backend_service.dataset_storage import DatasetWriter
from backend_service.failures import ApplicationFailure

Usage = namedtuple("Usage", "total used free")


def writer_for(tmp_path: Path) -> DatasetWriter:
    """Create separate source and output roots for one writer."""
    (tmp_path / "source").mkdir()
    return DatasetWriter(tmp_path / "output", tmp_path / "source")


def test_write_stream_hashes_counts_and_protects_the_stage(tmp_path: Path) -> None:
    """Chunks are charged as they arrive and the file is private and new."""
    with writer_for(tmp_path) as writer:
        digest, size = writer.write_stream(
            "images/a.bin", iter([b"ab", b"cd"]), maximum_bytes=10
        )
        assert (digest, size) == (hashlib.sha256(b"abcd").hexdigest(), 4)
        assert writer.authorized_bytes == 4
        path = writer.stage / "images" / "a.bin"
        assert stat.S_IMODE(os.lstat(path).st_mode) == 0o600
        with pytest.raises(ApplicationFailure) as existing:
            writer.write_stream("images/a.bin", iter([b"x"]), maximum_bytes=10)
        assert existing.value.code == "dataset_access"
        with pytest.raises(ApplicationFailure) as too_large:
            writer.write_stream("images/b.bin", iter([b"abc", b"def"]), maximum_bytes=5)
        assert too_large.value.code == "dataset_limits"
        with pytest.raises(ApplicationFailure):
            writer.write_stream("../escape.bin", iter([b"x"]), maximum_bytes=5)


def test_disk_reserve_refuses_writes_below_the_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writes stop before the free space drops under 10 GiB or 10 percent."""
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda path: Usage(200 * 1024**3, 0, 11 * 1024**3),
    )
    assert disk_reserve.minimum_free_bytes(tmp_path) == 20 * 1024**3
    assert disk_reserve.free_disk_bytes(tmp_path) == 11 * 1024**3
    with pytest.raises(ApplicationFailure) as failure:
        disk_reserve.ensure_disk_reserve(tmp_path, 1)
    assert failure.value.code == "dataset_space"
    with pytest.raises(ApplicationFailure):
        disk_reserve.ensure_disk_reserve(tmp_path, -1)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda path: Usage(200 * 1024**3, 0, 100 * 1024**3),
    )
    disk_reserve.ensure_disk_reserve(tmp_path, 1024)
    with writer_for(tmp_path) as writer:
        writer.write_stream("a.bin", iter([b"ok"]), maximum_bytes=2)
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda path: Usage(200 * 1024**3, 0, 19 * 1024**3),
        )
        with pytest.raises(ApplicationFailure) as failure:
            writer.write_stream("b.bin", iter([b"no"]), maximum_bytes=2)
        assert failure.value.code == "dataset_space"
