"""Hash determinism, variant robustness, bounded distances and tampering."""

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest
from near_duplicate_fixtures import (
    recompressed_variant,
    resized_variant,
    shifted_crop_variant,
    smooth_image,
    write_revision,
)
from numpy.typing import NDArray

from backend_service.dataset_serialization import contained_file
from backend_service.dataset_validation import validated_records
from backend_service.failures import ApplicationFailure
from backend_service.near_duplicate_hash import (
    METHOD_VERSION,
    dct_matrix,
    hamming_distances,
    hash_prepared_image,
    pairs_within,
    perceptual_hash,
)

REFERENCE_HASH = 0xB099F0A35892EE5E
UNRELATED_SEEDS = (0, 1, 2, 4, 6, 8)


def distance(first: int, second: int) -> int:
    """Count differing bits the slow way."""
    return (first ^ second).bit_count()


def close_triples(
    first: NDArray[np.uint64], second: NDArray[np.uint64], *, same_set: bool
) -> list[tuple[int, int, int]]:
    """List (first index, second index, distance) at or below 12 the slow way."""
    indices = (
        combinations(range(len(first)), 2)
        if same_set
        else product(range(len(first)), range(len(second)))
    )
    triples = [
        (
            first_index,
            second_index,
            distance(int(first[first_index]), int(second[second_index])),
        )
        for first_index, second_index in indices
    ]
    return [triple for triple in triples if triple[2] <= 12]


def random_hashes(seed: int, count: int) -> NDArray[np.uint64]:
    """Draw uniformly random 64-bit hashes."""
    generator = np.random.default_rng(seed)
    limit = int(np.iinfo(np.uint64).max)
    return generator.integers(0, limit, size=count, dtype=np.uint64, endpoint=True)


def test_dct_basis_is_orthonormal() -> None:
    """The precomputed basis must invert by its transpose."""
    basis = dct_matrix()
    assert basis.shape == (32, 32)
    assert np.allclose(basis @ basis.T, np.eye(32))


def test_reference_hash_guards_the_method() -> None:
    """A fixed synthetic picture keeps its recorded hash; copies hash equal."""
    assert METHOD_VERSION == "phash_dct_32_v1"
    assert perceptual_hash(smooth_image(7)) == REFERENCE_HASH
    assert perceptual_hash(smooth_image(7).copy()) == REFERENCE_HASH


def test_resized_and_recompressed_variants_stay_close() -> None:
    """Resolution and JPEG round trips move at most four bits."""
    for seed in range(6):
        image = smooth_image(seed)
        value = perceptual_hash(image)
        assert distance(value, perceptual_hash(resized_variant(image))) <= 4
        assert distance(value, perceptual_hash(recompressed_variant(image))) <= 4


def test_unrelated_and_shifted_images_are_far() -> None:
    """Unrelated seeds differ by twenty bits or more; a shifted crop is not close."""
    hashes = {seed: perceptual_hash(smooth_image(seed)) for seed in UNRELATED_SEEDS}
    for first, second in combinations(UNRELATED_SEEDS, 2):
        assert distance(hashes[first], hashes[second]) >= 20
    shifted = perceptual_hash(shifted_crop_variant(smooth_image(0)))
    assert distance(hashes[0], shifted) > 8


def test_hamming_distances_match_a_python_loop() -> None:
    """Blocked bit counting equals the slow loop across a block boundary."""
    first = random_hashes(1, 2050)
    second = random_hashes(2, 23)
    expected = [
        [distance(int(first_hash), int(second_hash)) for second_hash in second]
        for first_hash in first
    ]
    result = hamming_distances(first, second)
    assert result.dtype == np.uint8
    assert result.tolist() == expected


def test_pairs_within_lists_each_close_pair_once() -> None:
    """Planted near duplicates are the only close pairs and appear once each."""
    values = random_hashes(3, 300)
    values[7] = values[3] ^ np.uint64(0b101)
    values[250] = values[3]
    expected = close_triples(values, values, same_set=True)
    assert sorted(pairs_within(values, values, threshold=12, same_set=True)) == expected
    assert [pair[:2] for pair in expected] == [(3, 7), (3, 250), (7, 250)]
    others = random_hashes(4, 40)
    others[5] = values[7] ^ np.uint64(1)
    expected_cross = close_triples(values, others, same_set=False)
    found = sorted(pairs_within(values, others, threshold=12, same_set=False))
    assert found == expected_cross
    assert [pair[:2] for pair in found] == [(3, 5), (7, 5), (250, 5)]


def test_hash_prepared_image_refuses_a_tampered_file(tmp_path: Path) -> None:
    """Only bytes matching the frozen record are hashed."""
    revision = write_revision(
        tmp_path / "datasets" / "audit_fixture" / "staging",
        [("training/000.png", smooth_image(5), "training")],
    )
    _, records = validated_records(revision)
    record = records[0]
    path = contained_file(revision, record.prepared_path)
    assert hash_prepared_image(path, record) == perceptual_hash(smooth_image(5))
    wrong_record = record.model_copy(update={"prepared_checksum": "0" * 64})
    with pytest.raises(ApplicationFailure) as mismatch:
        hash_prepared_image(path, wrong_record)
    assert mismatch.value.code == "audit_integrity"
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ApplicationFailure) as tampered:
        hash_prepared_image(path, record)
    assert tampered.value.code == "audit_integrity"
    with pytest.raises(ApplicationFailure) as missing:
        hash_prepared_image(revision / "images" / "absent.png", record)
    assert missing.value.code == "audit_integrity"
