"""Streaming parquet readers for image bytes and text rows."""

import inspect
import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow
import pyarrow.parquet
import pytest
from PIL import Image

from backend_service.dataset_sources.parquet_reader import (
    ParquetColumns,
    ParquetImage,
    inspect_parquet_columns,
    iter_image_rows,
    iter_text_rows,
)
from backend_service.failures import ApplicationFailure

IMAGE_RECORD = pyarrow.struct([("bytes", pyarrow.binary()), ("path", pyarrow.string())])
TEXT_ROWS = ["plain ascii", "héllo wörld", "日本語のテキスト"]


def encoded_image(image_format: str, color: tuple[int, int, int]) -> bytes:
    """Return a real 2x2 image encoded by Pillow."""
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format=image_format)
    return buffer.getvalue()


IMAGES = [
    encoded_image("PNG", (255, 0, 0)),
    encoded_image("PNG", (0, 0, 255)),
    encoded_image("JPEG", (0, 255, 0)),
]
IMAGE_RECORDS = [
    {"bytes": data, "path": f"{index}.img"} for index, data in enumerate(IMAGES)
]


def write_parquet(path: Path, columns: dict[str, Any], **options: Any) -> Path:
    """Write one parquet file from arrow arrays and return its path."""
    pyarrow.parquet.write_table(pyarrow.table(columns), path, **options)
    return path


def shard(tmp_path: Path, **columns: Any) -> Path:
    """Write default-named image and text columns, one row per row group."""
    columns = {
        "image": pyarrow.array(IMAGE_RECORDS, type=IMAGE_RECORD),
        "text": pyarrow.array(TEXT_ROWS),
        **columns,
    }
    return write_parquet(tmp_path / "shard.parquet", columns, row_group_size=1)


def failure_code(iterator: Iterator[Any], collected: list[Any]) -> str:
    """Drain an iterator into ``collected`` and return the failure code raised."""
    with pytest.raises(ApplicationFailure) as raised:
        collected.extend(iterator)
    return raised.value.code


class BatchLog:
    """Record the batch sizes requested from pyarrow and the rows per batch."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrap ``ParquetFile.iter_batches`` for the duration of one test."""
        self.batch_sizes: list[int] = []
        self.rows_per_batch: list[int] = []
        original = pyarrow.parquet.ParquetFile.iter_batches

        def recording(parquet: Any, *arguments: Any, **options: Any) -> Iterator[Any]:
            """Log the request and every batch handed out."""
            self.batch_sizes.append(options["batch_size"])
            for batch in original(parquet, *arguments, **options):
                self.rows_per_batch.append(batch.num_rows)
                yield batch

        monkeypatch.setattr(pyarrow.parquet.ParquetFile, "iter_batches", recording)


def test_inspect_resolves_default_columns_and_row_groups(tmp_path: Path) -> None:
    """Default names resolve when present and the row layout is reported."""
    columns = inspect_parquet_columns(
        shard(tmp_path), image_column=None, text_column=None
    )
    assert columns == ParquetColumns("image", "text", row_count=3, row_groups=3)


def test_inspect_resolves_explicit_binary_and_text_columns(tmp_path: Path) -> None:
    """Explicit names accept a plain binary image column and a string column."""
    path = write_parquet(
        tmp_path / "custom.parquet",
        {
            "picture": pyarrow.array(IMAGES, type=pyarrow.binary()),
            "caption": pyarrow.array(TEXT_ROWS, type=pyarrow.large_string()),
        },
    )
    columns = inspect_parquet_columns(path, image_column="picture", text_column=None)
    assert columns == ParquetColumns("picture", None, row_count=3, row_groups=1)
    columns = inspect_parquet_columns(path, image_column=None, text_column="caption")
    assert columns.image_column is None and columns.text_column == "caption"
    with pytest.raises(ApplicationFailure) as raised:
        inspect_parquet_columns(path, image_column=None, text_column=None)
    assert (raised.value.code, raised.value.status_code) == ("source_schema", 422)


@pytest.mark.parametrize(
    ("image_column", "text_column"),
    [("missing", None), (None, "missing"), ("number", None), (None, "number")],
)
def test_inspect_rejects_missing_or_wrong_typed_explicit_columns(
    tmp_path: Path, image_column: str | None, text_column: str | None
) -> None:
    """An explicitly requested column must exist with the expected type."""
    path = shard(tmp_path, number=pyarrow.array([1, 2, 3]))
    with pytest.raises(ApplicationFailure) as raised:
        inspect_parquet_columns(
            path, image_column=image_column, text_column=text_column
        )
    assert (raised.value.code, raised.value.status_code) == ("source_schema", 422)


def test_inspect_treats_wrong_typed_default_columns_as_absent(tmp_path: Path) -> None:
    """A default name holding another kind of value is ignored, not fatal."""
    path = write_parquet(
        tmp_path / "addresses.parquet",
        {"image": pyarrow.array(["a.png", "b.png"]), "text": pyarrow.array([1, 2])},
    )
    columns = inspect_parquet_columns(path, image_column=None, text_column="image")
    assert columns == ParquetColumns(None, "image", row_count=2, row_groups=1)


def test_inspect_rejects_unreadable_files(tmp_path: Path) -> None:
    """Damaged and missing files fail with a plain message, not a crash."""
    damaged = tmp_path / "damaged.parquet"
    damaged.write_bytes(b"PAR1 this is not a parquet file")
    for path in (damaged, tmp_path / "absent.parquet"):
        with pytest.raises(ApplicationFailure) as raised:
            inspect_parquet_columns(path, image_column=None, text_column=None)
        assert raised.value.code == "source_unreadable"


def test_iter_image_rows_yields_every_row_across_row_groups(tmp_path: Path) -> None:
    """Rows come back in order with their bytes intact and their file hints."""
    rows = list(iter_image_rows(shard(tmp_path), "image", maximum_rows=10))
    assert rows == [
        ParquetImage(data=data, path_hint=f"{index}.img", row=index)
        for index, data in enumerate(IMAGES)
    ]
    formats = [Image.open(io.BytesIO(row.data)).format for row in rows]
    assert formats == ["PNG", "PNG", "JPEG"]


def test_iter_image_rows_reads_plain_binary_columns(tmp_path: Path) -> None:
    """A binary column yields the bytes with an empty file hint."""
    path = write_parquet(
        tmp_path / "binary.parquet",
        {"picture": pyarrow.array(IMAGES, type=pyarrow.binary())},
    )
    rows = list(iter_image_rows(path, "picture", maximum_rows=10))
    assert [(row.data, row.path_hint, row.row) for row in rows] == [
        (data, "", index) for index, data in enumerate(IMAGES)
    ]


def test_iter_image_rows_keeps_rows_without_bytes(tmp_path: Path) -> None:
    """Missing records or missing bytes are yielded empty rather than skipped."""
    records = [None, {"bytes": None, "path": "gone.png"}, IMAGE_RECORDS[0]]
    path = write_parquet(
        tmp_path / "gaps.parquet", {"image": pyarrow.array(records, type=IMAGE_RECORD)}
    )
    rows = list(iter_image_rows(path, "image", maximum_rows=10))
    assert rows == [
        ParquetImage(data=b"", path_hint="", row=0),
        ParquetImage(data=b"", path_hint="gone.png", row=1),
        ParquetImage(data=IMAGES[0], path_hint="0.img", row=2),
    ]


def test_iter_image_rows_honours_maximum_rows(tmp_path: Path) -> None:
    """The row cap stops iteration without reading further batches."""
    path = shard(tmp_path)
    assert [row.row for row in iter_image_rows(path, "image", maximum_rows=2)] == [0, 1]
    assert list(iter_image_rows(path, "image", maximum_rows=0)) == []


def test_iter_image_rows_rejects_rows_above_maximum_bytes(tmp_path: Path) -> None:
    """Rows within the limit are yielded; the first larger row raises."""
    limit = max(len(IMAGES[0]), len(IMAGES[1]))
    assert len(IMAGES[2]) > limit
    collected: list[ParquetImage] = []
    iterator = iter_image_rows(
        shard(tmp_path), "image", maximum_rows=10, maximum_bytes=limit
    )
    assert failure_code(iterator, collected) == "source_limits"
    assert [row.row for row in collected] == [0, 1]


def test_iter_image_rows_rejects_a_column_without_images(tmp_path: Path) -> None:
    """Reading a text column as images fails with the schema code."""
    assert (
        failure_code(iter_image_rows(shard(tmp_path), "text", maximum_rows=10), [])
        == "source_schema"
    )


def test_iter_image_rows_is_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is opened before the first ``next`` and only one batch per step."""
    absent = iter_image_rows(tmp_path / "absent.parquet", "image", maximum_rows=1)
    assert inspect.isgenerator(absent)
    with pytest.raises(ApplicationFailure) as raised:
        next(absent)
    assert raised.value.code == "source_unreadable"
    log = BatchLog(monkeypatch)
    rows = iter_image_rows(shard(tmp_path), "image", maximum_rows=10, batch_size=1)
    assert inspect.isgenerator(rows)
    assert log.batch_sizes == []
    assert next(rows).row == 0
    assert (log.batch_sizes, log.rows_per_batch) == ([1], [1])
    assert next(rows).row == 1
    assert log.rows_per_batch == [1, 1]
    rows.close()


def test_iter_image_rows_memory_stays_bounded_with_batch_size_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arrow memory in use stays flat and far below the file while iterating."""
    row_bytes = 1024**2
    path = write_parquet(
        tmp_path / "large.parquet",
        {"image": pyarrow.array([os.urandom(row_bytes) for _ in range(8)])},
        row_group_size=1,
        compression="none",
    )
    log = BatchLog(monkeypatch)
    baseline = pyarrow.total_allocated_bytes()
    in_use: list[int] = []
    sizes: list[int] = []
    for row in iter_image_rows(path, "image", maximum_rows=10, batch_size=1):
        sizes.append(len(row.data))
        in_use.append(pyarrow.total_allocated_bytes() - baseline)
    assert sizes == [row_bytes] * 8
    assert log.rows_per_batch == [1] * 8
    assert max(in_use) < 4 * row_bytes
    assert max(in_use) - in_use[0] <= 256 * 1024


def test_iter_text_rows_yields_rows_and_enforces_the_byte_limit(
    tmp_path: Path,
) -> None:
    """Text rows stream in order; the limit counts UTF-8 bytes across rows."""
    path = shard(tmp_path)
    assert list(iter_text_rows(path, "text", maximum_bytes=10**6)) == TEXT_ROWS
    total = sum(len(row.encode("utf-8")) for row in TEXT_ROWS)
    assert list(iter_text_rows(path, "text", maximum_bytes=total)) == TEXT_ROWS
    collected: list[str] = []
    iterator = iter_text_rows(path, "text", maximum_bytes=total - 1, batch_size=1)
    assert failure_code(iterator, collected) == "source_limits"
    assert collected == TEXT_ROWS[:2]


def test_iter_text_rows_keeps_missing_values_and_rejects_other_types(
    tmp_path: Path,
) -> None:
    """Missing text becomes an empty row; non-text columns fail as schema."""
    path = write_parquet(
        tmp_path / "gaps.parquet",
        {
            "text": pyarrow.array(["first", None, "third"]),
            "number": pyarrow.array([1, 2, 3]),
        },
    )
    rows = list(iter_text_rows(path, "text", maximum_bytes=100))
    assert rows == ["first", "", "third"]
    assert failure_code(iter_text_rows(path, "number", maximum_bytes=100), []) == (
        "source_schema"
    )
    assert (
        failure_code(iter_text_rows(shard(tmp_path), "image", maximum_bytes=100), [])
        == "source_schema"
    )
