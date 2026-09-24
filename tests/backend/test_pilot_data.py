"""Verify deterministic pilot choices, lazy integrity, and exact crop resume."""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from PIL import Image
from pilot_data_fixtures import pilot_revision

from backend_service.dataset_serialization import canonical_json
from backend_service.failures import ApplicationFailure
from backend_service.pilot_data import FULL_UHD_REVISION, load_pilot_data
from backend_service.pilot_sampler import PilotSampler


@pytest.fixture(autouse=True)
def bounded_threads() -> Iterator[None]:
    """Keep small CPU tensor copies from using a large host thread pool."""
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_selection_is_metadata_only_and_hash_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Freeze 32 native covers and exclusions without decoding any image."""
    directory = pilot_revision(tmp_path / "stage", tuning_count=34)

    def forbid_pixels(*arguments: object, **keywords: object) -> None:
        """Reject eager access to training, tuning, or held-out pixels."""
        raise AssertionError("Metadata preparation must not decode pixels.")

    monkeypatch.setattr(Image, "open", forbid_pixels)
    data = load_pilot_data(directory)
    selected = data.selection
    assert len(data.training_examples) == 5
    assert len(data.tuning_examples) == 32
    assert selected.tuning_eligible_count == 34
    assert len(selected.tuning_excluded) == 2
    assert selected.split_counts == {"train": 5, "tuning": 36, "held_out": 1}
    identities = [f"fixture:{number}" for number in range(5, 39)]
    expected = sorted(
        identities,
        key=lambda identity: hashlib.sha256(
            canonical_json(["pilot_data_v1", data.manifest.revision, identity])
        ).hexdigest(),
    )[:32]
    assert [cover.source_identity for cover in selected.tuning_selected] == expected
    assert selected == load_pilot_data(directory).selection
    assert not selected.gpu_profile_eligible and selected.gpu_profile_reasons
    assert not selected.pilot_ready and not data.manifest.pilot_ready


@pytest.mark.parametrize("batch_size", [4, 8])
def test_resume_preserves_order_crops_and_consumed_position(
    tmp_path: Path, batch_size: int
) -> None:
    """A JSON snapshot reproduces multiple batches across epoch boundaries."""
    data = load_pilot_data(pilot_revision(tmp_path / "stage"))
    sampler = PilotSampler(data, 71)
    sampler.next_batch(batch_size)
    restored = PilotSampler(data, 71)
    restored.load_state_dict(json.loads(json.dumps(sampler.state_dict())))
    for _ in range(3):
        actual = sampler.next_batch(batch_size)
        expected = restored.next_batch(batch_size)
        assert torch.equal(actual, expected)
        assert sampler.last_batch == restored.last_batch
        assert sampler.state_dict() == restored.state_dict()
    assert sampler.consumed == batch_size * 4


def test_batch_partition_consumes_same_sixteen_samples(tmp_path: Path) -> None:
    """Physical batches of four and eight consume the same image/crop stream."""
    data = load_pilot_data(pilot_revision(tmp_path / "stage"))
    four, eight = PilotSampler(data, 2), PilotSampler(data, 2)
    tensor_four = torch.cat([four.next_batch(4) for _ in range(4)])
    tensor_eight = torch.cat([eight.next_batch(8) for _ in range(2)])
    assert torch.equal(tensor_four, tensor_eight)
    assert four.consumed == eight.consumed == 16
    assert four.state_dict() == eight.state_dict()


def test_lazy_checks_do_not_open_held_out_and_failed_batch_rolls_back(
    tmp_path: Path,
) -> None:
    """Corrupted held-out bytes are untouched; training corruption fails safely."""
    data = load_pilot_data(pilot_revision(tmp_path / "stage"))
    (data.training_examples[0].path.parent / "9.png").write_bytes(b"held out")
    sampler = PilotSampler(data, 14)
    sampler.next_batch(4)
    before = sampler.state_dict()
    before_batch = sampler.last_batch
    for example in data.training_examples:
        example.path.write_bytes(b"changed")
    with pytest.raises(ApplicationFailure) as caught:
        sampler.next_batch(8)
    assert caught.value.code == "pilot_dataset_invalid"
    assert sampler.state_dict() == before and sampler.last_batch == before_batch


@pytest.mark.parametrize("change", ["seed", "position", "order", "revision"])
def test_incompatible_sampler_snapshot_preserves_live_state(
    tmp_path: Path, change: str
) -> None:
    """Bad indices or provenance must not partly restore a live sampler."""
    data = load_pilot_data(pilot_revision(tmp_path / "stage"))
    sampler = PilotSampler(data, 9)
    before = sampler.state_dict()
    changed = dict(before)
    if change == "seed":
        changed["seed"] = 10
    elif change == "position":
        changed["position"] = 6
    elif change == "order":
        changed["order"] = [0] * 5
    else:
        changed["dataset_revision"] = "0" * 64
    with pytest.raises(ApplicationFailure) as caught:
        sampler.load_state_dict(changed)
    assert caught.value.code == "pilot_sampler_invalid"
    assert sampler.state_dict() == before


def test_record_corruption_and_deadline_are_clear(tmp_path: Path) -> None:
    """Reject changed metadata and preserve deadline identity before any reads."""
    directory = pilot_revision(tmp_path / "stage")
    with pytest.raises(ApplicationFailure) as caught:
        load_pilot_data(directory, deadline=0)
    assert caught.value.code == "pilot_deadline"
    (directory / "records.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ApplicationFailure) as caught:
        load_pilot_data(directory)
    assert caught.value.code == "pilot_dataset_invalid"


def test_existing_full_revision_selection_without_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the real revision when installed without reading its held-out pixels."""
    directory = Path("datasets/uhd_iqa") / FULL_UHD_REVISION
    if not directory.is_dir():
        pytest.skip("The immutable full UHD-IQA revision is not installed.")

    def forbid_pixels(*arguments: object, **keywords: object) -> None:
        """Enforce the corpus metadata-only inspection boundary."""
        raise AssertionError("Full dataset preparation must not decode images.")

    monkeypatch.setattr(Image, "open", forbid_pixels)
    data = load_pilot_data(directory)
    assert data.selection.gpu_profile_eligible
    assert data.selection.split_counts == {
        "train": 4269,
        "tuning": 904,
        "held_out": 900,
    }
    assert len(data.training_examples) == 4269 and len(data.tuning_examples) == 32
    assert data.selection.tuning_eligible_count == 902
    assert len(data.selection.tuning_excluded) == 2
