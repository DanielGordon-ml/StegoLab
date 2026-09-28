"""Prove the frozen benchmark identities are deterministic, ordered and guarded."""

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from backend_service.benchmark_freeze import (
    freeze_benchmark,
    read_annotation_set,
    selection_key,
)
from backend_service.benchmark_validation import (
    load_benchmark_identities,
    validate_benchmark,
)
from backend_service.failures import ApplicationFailure
from schemas.benchmark_identities import BenchmarkFreezeRequest, BenchmarkSourceArchive

VALIDATION_IDENTIFIERS = [100, 101, 102, 103, 104, 105]
TRAINING_IDENTIFIERS = list(range(1000, 1040))
SEED = 7
ARCHIVES = [
    BenchmarkSourceArchive(
        repository="pcuenq/coco-2017-mirror",
        revision="a4cd8b69bd45a35e9a0a9c8692f3a7f4f23321fe",
        archive_path=name,
        archive_sha256="0" * 64,
        archive_bytes=size,
    )
    for name, size in (("val2017.zip", 815585330), ("train2017.zip", 19336861798))
]


def _canonical(value: object) -> bytes:
    """Re-implement canonical JSON independently of the backend helper."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return (payload + "\n").encode("utf-8")


def _images(identifiers: list[int], extension: str = "jpg") -> list[dict[str, object]]:
    """Build COCO-style image entries with varied declared sizes."""
    return [
        {
            "id": identifier,
            "file_name": f"{identifier:012d}.{extension}",
            "width": 640,
            "height": 480 + identifier % 7,
            "license": 1,
        }
        for identifier in identifiers
    ]


def _write_annotation(
    directory: Path, name: str, images: list[dict[str, object]]
) -> None:
    """Write one annotation file with an images array and unrelated content."""
    directory.mkdir(parents=True, exist_ok=True)
    document = {"info": {"year": 2017}, "images": images, "annotations": []}
    (directory / name).write_text(json.dumps(document), encoding="utf-8")


def write_annotations(directory: Path) -> Path:
    """Write the six validation and forty training entries as three files."""
    validation = _images(VALIDATION_IDENTIFIERS)
    _write_annotation(directory, "captions_val2017.json", validation)
    _write_annotation(directory, "instances_val2017.json", list(reversed(validation)))
    _write_annotation(
        directory, "captions_train2017.json", _images(TRAINING_IDENTIFIERS)
    )
    return directory


def freeze_request(
    annotations: Path, output: Path, **changes: object
) -> BenchmarkFreezeRequest:
    """Build the synthetic freeze request with optional field overrides."""
    values: dict[str, object] = {
        "annotations_directory": str(annotations),
        "output_directory": str(output),
        "source_archives": ARCHIVES,
        "seed": SEED,
        "training_source_count": 10,
        "reserve_count": 5,
        "expect_official_counts": False,
    }
    return BenchmarkFreezeRequest.model_validate(values | changes)


def reference_training_order(seed: int) -> list[str]:
    """Order training member paths by an independent copy of the selection rule."""
    paths = [f"train2017/{identifier:012d}.jpg" for identifier in TRAINING_IDENTIFIERS]
    return sorted(
        paths,
        key=lambda path: (
            hashlib.sha256(
                _canonical(["release_benchmark_v1", seed, path])
            ).hexdigest(),
            path,
        ),
    )


def _failure_code(function: Callable[..., object], *arguments: object) -> str:
    """Return the failure code raised by calling the function."""
    with pytest.raises(ApplicationFailure) as caught:
        function(*arguments)
    return caught.value.code


def _rewrite(directory: Path, lines: list[bytes], header: dict[str, object]) -> None:
    """Rewrite both files with checksums recomputed by the test's own rule."""
    header["members_checksum"] = hashlib.sha256(b"".join(lines)).hexdigest()
    header.pop("identities_checksum")
    header["identities_checksum"] = hashlib.sha256(_canonical(header)).hexdigest()
    (directory / "members.jsonl").write_bytes(b"".join(lines))
    (directory / "identities.json").write_bytes(_canonical(header))


def test_freeze_is_deterministic_and_ranks_follow_the_selection_key(
    tmp_path: Path,
) -> None:
    """Two freezes write identical bytes and rank training members by key."""
    annotations = write_annotations(tmp_path / "annotations")
    first, second = tmp_path / "first", tmp_path / "second"
    header = freeze_benchmark(freeze_request(annotations, first))
    assert freeze_benchmark(freeze_request(annotations, second)) == header
    for name in ("identities.json", "members.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first / "identities.json").read_bytes() == _canonical(
        header.model_dump(mode="json")
    )
    assert (header.validation_count, header.training_source_count) == (6, 10)
    assert (header.reserve_count, header.member_count) == (5, 21)
    assert header.pilot_ready is False and header.preparation_policy == "not_frozen"
    assert [file.image_count for file in header.annotation_files] == [6, 6, 40]
    loaded, members = load_benchmark_identities(first)
    assert loaded == header
    validation_paths = [
        f"val2017/{number:012d}.jpg" for number in VALIDATION_IDENTIFIERS
    ]
    assert [member.member_path for member in members[:6]] == validation_paths
    expected = reference_training_order(SEED)[:15]
    assert [member.member_path for member in members[6:]] == expected
    assert [member.rank for member in members[6:]] == list(range(15))
    roles = [member.role for member in members[6:]]
    assert roles == ["training_source"] * 10 + ["reserve"] * 5
    assert (
        selection_key("release_benchmark_v1", SEED, expected[0])
        == hashlib.sha256(
            _canonical(["release_benchmark_v1", SEED, expected[0]])
        ).hexdigest()
    )
    assert reference_training_order(SEED + 1)[:15] != expected


def test_freeze_refuses_bad_inputs_without_creating_output(tmp_path: Path) -> None:
    """Existing output, disagreeing or malformed annotations and small pools fail."""
    annotations = write_annotations(tmp_path / "annotations")
    output = tmp_path / "frozen"
    freeze_benchmark(freeze_request(annotations, output))
    request = freeze_request(annotations, output)
    assert _failure_code(freeze_benchmark, request) == "benchmark_exists"
    small = freeze_request(annotations, tmp_path / "small", training_source_count=36)
    assert _failure_code(freeze_benchmark, small) == "benchmark_population"
    official = freeze_request(
        annotations, tmp_path / "official", expect_official_counts=True
    )
    assert _failure_code(freeze_benchmark, official) == "benchmark_annotations"
    cases = {
        "disagreement": ("instances_val2017.json", _images(VALIDATION_IDENTIFIERS[:5])),
        "duplicate": (
            "captions_train2017.json",
            _images(TRAINING_IDENTIFIERS + [1000]),
        ),
        "file_name": ("captions_train2017.json", _images(TRAINING_IDENTIFIERS, "png")),
    }
    for label, (name, images) in cases.items():
        broken = write_annotations(tmp_path / label)
        _write_annotation(broken, name, images)
        request = freeze_request(broken, tmp_path / f"{label}_output")
        assert _failure_code(freeze_benchmark, request) == "benchmark_annotations"
    missing = freeze_request(tmp_path / "absent", tmp_path / "absent_output")
    assert _failure_code(freeze_benchmark, missing) == "benchmark_annotations"
    assert (
        sorted(path.name for path in tmp_path.iterdir() if "output" in path.name) == []
    )
    assert not (tmp_path / "small").exists() and not (tmp_path / "official").exists()


def test_load_detects_tampered_member_line_and_header_field(tmp_path: Path) -> None:
    """Changing a member, a header field or the key order is refused."""
    annotations = write_annotations(tmp_path / "annotations")
    original = tmp_path / "frozen"
    freeze_benchmark(freeze_request(annotations, original))
    lines = (original / "members.jsonl").read_bytes().splitlines(keepends=True)
    header: dict[str, object] = json.loads((original / "identities.json").read_bytes())
    untouched = tmp_path / "untouched"
    shutil.copytree(original, untouched)
    _rewrite(untouched, list(lines), dict(header))
    loaded, _ = load_benchmark_identities(untouched)
    assert loaded.identities_checksum == header["identities_checksum"]
    member_tampered = tmp_path / "member"
    shutil.copytree(original, member_tampered)
    member = json.loads(lines[8])
    member["declared_width"] += 1
    changed = lines[:8] + [_canonical(member)] + lines[9:]
    (member_tampered / "members.jsonl").write_bytes(b"".join(changed))
    assert (
        _failure_code(load_benchmark_identities, member_tampered) == "benchmark_invalid"
    )
    header_tampered = tmp_path / "header"
    shutil.copytree(original, header_tampered)
    forged = json.loads(json.dumps(header))
    forged["selection_rule"]["seed"] = SEED + 1
    (header_tampered / "identities.json").write_bytes(_canonical(forged))
    assert (
        _failure_code(load_benchmark_identities, header_tampered) == "benchmark_invalid"
    )
    reordered = tmp_path / "reordered"
    shutil.copytree(original, reordered)
    first, second = json.loads(lines[6]), json.loads(lines[7])
    first["rank"], second["rank"] = 1, 0
    swapped = lines[:6] + [_canonical(second), _canonical(first)] + lines[8:]
    _rewrite(reordered, swapped, dict(header))
    assert _failure_code(load_benchmark_identities, reordered) == "benchmark_invalid"
    assert _failure_code(load_benchmark_identities, tmp_path / "absent") == (
        "benchmark_invalid"
    )


def test_validate_benchmark_with_and_without_annotations(tmp_path: Path) -> None:
    """Validation reports the frozen counts and re-derives from annotations."""
    annotations = write_annotations(tmp_path / "annotations")
    output = tmp_path / "frozen"
    header = freeze_benchmark(freeze_request(annotations, output))
    report = validate_benchmark(output)
    assert report.integrity == "verified" and report.pilot_ready is False
    assert report.identities_checksum == header.identities_checksum
    assert report.members_checksum == header.members_checksum
    assert (report.validation_count, report.member_count) == (6, 21)
    with_annotations = validate_benchmark(output, annotations)
    assert with_annotations.integrity == "verified_with_annotations"
    assert with_annotations.identities_checksum == header.identities_checksum
    listing = sorted(path.name for path in output.iterdir())
    assert listing == ["identities.json", "members.jsonl"]
    other = write_annotations(tmp_path / "other")
    _write_annotation(
        other, "captions_train2017.json", _images(list(range(1000, 1041)))
    )
    assert _failure_code(validate_benchmark, output, other) == "benchmark_mismatch"
    assert _failure_code(validate_benchmark, output, tmp_path / "absent") == (
        "benchmark_annotations"
    )


def test_symlinked_ancestors_are_benchmark_failures(tmp_path: Path) -> None:
    """Paths reached through a symbolic link fail with the benchmark codes."""
    annotations = write_annotations(tmp_path / "real" / "annotations")
    frozen = tmp_path / "real" / "frozen"
    freeze_benchmark(freeze_request(annotations, frozen))
    linked = tmp_path / "link"
    linked.symlink_to(tmp_path / "real", target_is_directory=True)
    assert _failure_code(
        read_annotation_set, linked / "annotations", "captions_val2017.json"
    ) == ("benchmark_annotations")
    assert _failure_code(load_benchmark_identities, linked / "frozen") == (
        "benchmark_invalid"
    )
    request = freeze_request(linked / "annotations", tmp_path / "linked_output")
    assert _failure_code(freeze_benchmark, request) == "benchmark_annotations"
    assert _failure_code(validate_benchmark, linked / "frozen") == "benchmark_invalid"
    assert not (tmp_path / "linked_output").exists()
