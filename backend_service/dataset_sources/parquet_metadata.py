"""Read parquet sizes from metadata and arrow arrays without copying any row.

``pyarrow`` is imported lazily inside each function so the rest of the
backend starts without it.
"""

from typing import Any


def largest_row_group_bytes(parquet: Any, column: str) -> int:
    """Return the most uncompressed bytes any row group holds for one column.

    ``pyarrow`` decodes a whole row group at a time, so this figure bounds the
    memory one batch of the column needs. A struct column counts every leaf
    beneath it. Only the file metadata is read.
    """
    metadata = parquet.metadata
    largest = 0
    for index in range(int(metadata.num_row_groups)):
        group = metadata.row_group(index)
        total = 0
        for position in range(int(group.num_columns)):
            chunk = group.column(position)
            name = str(chunk.path_in_schema)
            if name == column or name.startswith(column + "."):
                total += int(chunk.total_uncompressed_size)
        largest = max(largest, total)
    return largest


def byte_lengths(values: Any) -> Any:
    """Return the byte length of every row's image bytes without copying them.

    The result is an arrow integer array with a missing value for every row
    whose bytes are missing.
    """
    import pyarrow.compute
    import pyarrow.types

    data = values.field("bytes") if pyarrow.types.is_struct(values.type) else values
    return pyarrow.compute.binary_length(data)
