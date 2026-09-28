"""Perceptual DCT hashes and bounded Hamming distances for near-duplicate audits."""

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from backend_service.dataset_serialization import checksum, read_bounded
from backend_service.failures import ApplicationFailure
from schemas.datasets import DatasetImageRecord
from schemas.near_duplicate_audit import AUDIT_THRESHOLDS

METHOD_VERSION = "phash_dct_32_v1"
HASH_SIDE = 32
LOW_FREQUENCY_SIDE = 8
BLOCK_ROWS = 2048
MAXIMUM_PREPARED_BYTES = 128 * 1024**2
MAXIMUM_THRESHOLD = max(AUDIT_THRESHOLDS)
AUDIT_MESSAGES = {
    "audit_integrity": "A prepared image changed since its revision was written. "
    "Validate the dataset before auditing.",
    "audit_dataset_invalid": "The dataset revision is missing or invalid. "
    "Validate the dataset before auditing.",
    "audit_benchmark_dataset_invalid": "The benchmark dataset revision is missing "
    "or invalid. Validate it before auditing.",
    "audit_same_revision": "The benchmark revision is the audited revision. Point "
    "benchmark_dataset_directory at a different prepared revision.",
    "audit_benchmark_identities_required": "A benchmark revision needs the frozen "
    "identities. Pass the folder written by `stegolab freeze_benchmark` as "
    "benchmark_identities_directory.",
    "audit_benchmark_unmatched": "No record of the benchmark revision names a "
    "frozen member. Prepare the benchmark from the COCO archive so its records "
    "carry member paths such as val2017/<identifier>.jpg.",
    "audit_report_invalid": "The audit report is missing, unreadable or changed "
    "since it was written. Run the audit again.",
    "audit_output_overlaps": "The output root is a dataset revision or a dataset "
    "name folder. Choose a root beside the dataset folders; the default is the "
    "folder that holds them.",
    "audit_output_exists": "An audit folder with this identifier already exists. "
    "Run the audit again after one second or choose another output root.",
    "audit_storage": "The audit report could not be saved. Check access, symbolic "
    "links and free space under the output root.",
}


def audit_failure(code: str) -> ApplicationFailure:
    """Return a fixed audit failure without paths, values or parser details."""
    return ApplicationFailure(
        code, AUDIT_MESSAGES[code], 503 if code == "audit_storage" else 422
    )


def dct_matrix(size: int = HASH_SIDE) -> NDArray[np.float64]:
    """Return the orthonormal DCT-II basis whose rows are the frequencies."""
    positions = np.arange(size, dtype=np.float64)
    angles = np.pi * (2.0 * positions[None, :] + 1.0) * positions[:, None] / (2 * size)
    basis: NDArray[np.float64] = np.cos(angles) * np.sqrt(2.0 / size)
    basis[0, :] = np.sqrt(1.0 / size)
    return basis


DCT_BASIS = dct_matrix()


def perceptual_hash(image: Image.Image) -> int:
    """Hash the low frequencies of a 32×32 grey copy as 64 bits, first row first."""
    with image.convert("L") as grey:
        with grey.resize((HASH_SIDE, HASH_SIDE), Image.Resampling.BOX) as small:
            pixels = np.asarray(small, dtype=np.float64)
    coefficients = DCT_BASIS @ pixels @ DCT_BASIS.T
    block = np.round(coefficients[:LOW_FREQUENCY_SIDE, :LOW_FREQUENCY_SIDE], 6)
    bits = (block > np.median(block)).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def hash_prepared_image(path: Path, record: DatasetImageRecord) -> int:
    """Hash one prepared image after proving its bytes are the frozen ones."""
    try:
        data = read_bounded(path, MAXIMUM_PREPARED_BYTES)
    except (OSError, ValueError, ApplicationFailure):
        raise audit_failure("audit_integrity") from None
    if (checksum(data), len(data)) != (record.prepared_checksum, record.prepared_bytes):
        raise audit_failure("audit_integrity")
    try:
        with Image.open(BytesIO(data), formats=["PNG"]) as opened:
            if opened.size != (record.width, record.height):
                raise ValueError("Prepared dimensions differ from the record.")
            opened.load()
            return perceptual_hash(opened)
    except (OSError, ValueError, SyntaxError, TypeError, Image.DecompressionBombError):
        raise audit_failure("audit_integrity") from None


def hamming_distances(
    first: NDArray[np.uint64], second: NDArray[np.uint64]
) -> NDArray[np.uint8]:
    """Count differing bits for every pair, one block of 2,048 rows at a time."""
    distances = np.empty((len(first), len(second)), dtype=np.uint8)
    for start in range(0, len(first), BLOCK_ROWS):
        rows = first[start : start + BLOCK_ROWS]
        distances[start : start + BLOCK_ROWS] = np.bitwise_count(
            rows[:, None] ^ second[None, :]
        )
    return distances


def pairs_within(
    first: NDArray[np.uint64],
    second: NDArray[np.uint64],
    *,
    threshold: int,
    same_set: bool,
) -> Iterator[tuple[int, int, int]]:
    """Yield (first index, second index, distance) at or below the threshold.

    With ``same_set`` both arrays are the same hashes and every pair appears
    once with the smaller index first.
    """
    for row_start in range(0, len(first), BLOCK_ROWS):
        rows = first[row_start : row_start + BLOCK_ROWS]
        first_column = row_start if same_set else 0
        for column_start in range(first_column, len(second), BLOCK_ROWS):
            columns = second[column_start : column_start + BLOCK_ROWS]
            distances = hamming_distances(rows, columns)
            close = distances <= threshold
            if same_set and column_start == row_start:
                close &= np.triu(np.ones(close.shape, dtype=bool), k=1)
            for row, column in zip(*np.nonzero(close), strict=True):
                yield (
                    row_start + int(row),
                    column_start + int(column),
                    int(distances[row, column]),
                )
