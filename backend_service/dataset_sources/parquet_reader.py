"""Read image bytes and text rows from parquet files one batch at a time.

The readers never load a whole file. ``pyarrow`` decodes one row group at a
time, so memory stays bounded by the largest row group plus one row; the file
metadata is checked first so a row group larger than the fixed bound is
refused before anything is decoded. ``pyarrow.parquet`` is imported lazily so
the rest of the backend starts without it.
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend_service.dataset_sources.parquet_metadata import (
    byte_lengths,
    largest_row_group_bytes,
)
from backend_service.failures import ApplicationFailure

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_COLUMN = "image"
DEFAULT_TEXT_COLUMN = "text"
IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 256
DEFAULT_MAXIMUM_IMAGE_BYTES = 50 * 1024**2
MAXIMUM_ROW_GROUP_BYTES = 512 * 1024**2
READ_BUFFER_BYTES = 64 * 1024
IMAGE_RECORD_FIELDS = frozenset({"bytes", "path"})
_MESSAGES = {
    "source_schema": (
        "The parquet file does not have the expected image or text column, "
        "or that column holds the wrong kind of values. Image columns must "
        "hold raw image bytes or a Hugging Face image record; text columns "
        "must hold text. Check the column names in the request."
    ),
    "source_limits": (
        "A parquet row or row group is larger than the allowed size, or the "
        "text rows together exceed the allowed size. Select a smaller subset "
        "with explicit file names."
    ),
    "source_unreadable": (
        "The parquet file could not be read. It may be incomplete or damaged; "
        "clear the cached download and fetch it again."
    ),
}


def parquet_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for a parquet reading failure code."""
    return ApplicationFailure(code, _MESSAGES[code], 422)


@dataclass(frozen=True)
class ParquetColumns:
    """Column names resolved for one parquet file plus its row layout."""

    image_column: str | None
    text_column: str | None
    row_count: int
    row_groups: int


@dataclass(frozen=True)
class ParquetImage:
    """One image row: raw bytes, the upstream file name hint and the row index."""

    data: bytes
    path_hint: str
    row: int


def _open_parquet(path: Path) -> Any:
    """Open one parquet file for streaming reads with a small read buffer."""
    import pyarrow
    import pyarrow.parquet

    try:
        return pyarrow.parquet.ParquetFile(
            path, pre_buffer=False, buffer_size=READ_BUFFER_BYTES
        )
    except (OSError, pyarrow.ArrowException):
        logger.error("parquet_file_unreadable")
        raise parquet_failure("source_unreadable") from None


def _is_binary_type(data_type: Any) -> bool:
    """Tell whether an arrow type holds raw bytes."""
    import pyarrow.types

    return bool(
        pyarrow.types.is_binary(data_type) or pyarrow.types.is_large_binary(data_type)
    )


def _is_text_type(data_type: Any) -> bool:
    """Tell whether an arrow type holds text."""
    import pyarrow.types

    return bool(
        pyarrow.types.is_string(data_type) or pyarrow.types.is_large_string(data_type)
    )


def _is_image_type(data_type: Any) -> bool:
    """Accept raw bytes or the Hugging Face image record ``{bytes, path}``."""
    import pyarrow.types

    if _is_binary_type(data_type):
        return True
    if not pyarrow.types.is_struct(data_type):
        return False
    fields = {
        data_type.field(index).name: data_type.field(index).type
        for index in range(data_type.num_fields)
    }
    return (
        set(fields) == IMAGE_RECORD_FIELDS
        and _is_binary_type(fields["bytes"])
        and _is_text_type(fields["path"])
    )


def _resolve_column(
    schema: Any,
    requested: str | None,
    default: str,
    accepts: Callable[[Any], bool],
) -> str | None:
    """Return the usable column name; only an explicit request may fail.

    A default name is used when it is present with the right type and
    otherwise treated as absent, because a file may reuse the default name
    for something else (a text dataset with an ``image`` address column).
    """
    name = default if requested is None else requested
    index = schema.get_field_index(name)
    if index < 0:
        if requested is None:
            return None
        logger.error("parquet_column_missing")
        raise parquet_failure("source_schema")
    if not accepts(schema.field(index).type):
        if requested is None:
            logger.info("parquet_default_column_type_ignored")
            return None
        logger.error("parquet_column_type_mismatch")
        raise parquet_failure("source_schema")
    return name


def inspect_parquet_columns(
    path: Path, *, image_column: str | None, text_column: str | None
) -> ParquetColumns:
    """Resolve the image and text columns of one file without reading rows.

    An explicit column must exist as ``struct<bytes: binary, path: string>``
    or ``binary`` (images) or as a string column (text). When a name is
    ``None`` the default ``image`` / ``text`` column is used if usable. A file
    with no usable column at all is rejected with ``source_schema``.
    """
    parquet = _open_parquet(path)
    try:
        schema = parquet.schema_arrow
        image_name = _resolve_column(
            schema, image_column, DEFAULT_IMAGE_COLUMN, _is_image_type
        )
        text_name = _resolve_column(
            schema, text_column, DEFAULT_TEXT_COLUMN, _is_text_type
        )
        columns = ParquetColumns(
            image_column=image_name,
            text_column=text_name,
            row_count=int(parquet.metadata.num_rows),
            row_groups=int(parquet.metadata.num_row_groups),
        )
    finally:
        parquet.close()
    if image_name is None and text_name is None:
        logger.error("parquet_columns_unusable")
        raise parquet_failure("source_schema")
    logger.info("parquet_columns_inspected")
    return columns


def _check_row_groups(parquet: Any, column: str) -> None:
    """Refuse a file whose row groups hold more of the column than allowed."""
    if largest_row_group_bytes(parquet, column) > MAXIMUM_ROW_GROUP_BYTES:
        logger.error("parquet_row_group_too_large")
        raise parquet_failure("source_limits")


def _iter_column_batches(parquet: Any, column: str, batch_size: int) -> Iterator[Any]:
    """Yield the arrow values of one column batch by batch, wrapping read errors."""
    import pyarrow

    try:
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=[column], use_threads=False
        ):
            yield batch.column(0)
    except (OSError, pyarrow.ArrowException):
        logger.error("parquet_file_unreadable")
        raise parquet_failure("source_unreadable") from None


def _image_parts(value: Any) -> tuple[bytes, str]:
    """Split one decoded row into its bytes and its file name hint."""
    if value is None:
        return b"", ""
    if isinstance(value, dict):
        data = value.get("bytes")
        hint = value.get("path")
        return b"" if data is None else bytes(data), "" if hint is None else str(hint)
    return bytes(value), ""


def iter_image_rows(
    path: Path,
    column: str,
    *,
    maximum_rows: int,
    maximum_bytes: int = DEFAULT_MAXIMUM_IMAGE_BYTES,
    batch_size: int = IMAGE_BATCH_SIZE,
) -> Iterator[ParquetImage]:
    """Yield image rows lazily, in file order, never skipping a row.

    Every row up to ``maximum_rows`` is yielded as-is, including rows whose
    bytes are missing (``data`` is empty) so the caller can record a
    rejection; the caller applies its own image policy to the bytes. A row
    larger than ``maximum_bytes`` raises ``ApplicationFailure("source_limits")``
    after the rows before it were yielded and before its bytes are copied.
    Only ``batch_size`` rows are decoded at once, one row is copied at a time,
    and the file is opened on the first ``next()``.
    """
    parquet = _open_parquet(path)
    try:
        _resolve_column(parquet.schema_arrow, column, column, _is_image_type)
        _check_row_groups(parquet, column)
        row = 0
        for values in _iter_column_batches(parquet, column, batch_size):
            lengths = byte_lengths(values)
            for index in range(len(values)):
                if row >= maximum_rows:
                    return
                length = lengths[index].as_py()
                if length is not None and length > maximum_bytes:
                    logger.error("parquet_image_row_too_large")
                    raise parquet_failure("source_limits")
                data, path_hint = _image_parts(values[index].as_py())
                yield ParquetImage(data=data, path_hint=path_hint, row=row)
                row += 1
            if row >= maximum_rows:
                return
    finally:
        parquet.close()


def iter_text_rows(
    path: Path,
    column: str,
    *,
    maximum_bytes: int,
    batch_size: int = TEXT_BATCH_SIZE,
) -> Iterator[str]:
    """Yield text rows lazily in file order; missing values become empty strings.

    Raises ``ApplicationFailure("source_limits")`` as soon as the UTF-8 size of
    the rows yielded so far plus the next row would exceed ``maximum_bytes``;
    the rows before it were already yielded. The file is opened on the first
    ``next()`` and only ``batch_size`` rows are decoded at once.
    """
    parquet = _open_parquet(path)
    try:
        _resolve_column(parquet.schema_arrow, column, column, _is_text_type)
        _check_row_groups(parquet, column)
        total_bytes = 0
        for values in _iter_column_batches(parquet, column, batch_size):
            for value in values.to_pylist():
                text = "" if value is None else str(value)
                total_bytes += len(text.encode("utf-8"))
                if total_bytes > maximum_bytes:
                    logger.error("parquet_text_limit_exceeded")
                    raise parquet_failure("source_limits")
                yield text
    finally:
        parquet.close()
