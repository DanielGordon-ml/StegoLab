"""Persistent staging budgets, writer locking, publication, and cleanup."""

import json
import os
import shutil
from pathlib import Path

import pytest

from backend_service.dataset_storage import DatasetWriter
from backend_service.failures import ApplicationFailure


@pytest.fixture
def directories(tmp_path: Path) -> tuple[Path, Path]:
    """Separate read-only source semantics from disposable prepared outputs."""
    source = tmp_path / "source"
    source.mkdir()
    return source, tmp_path / "datasets"


@pytest.fixture(autouse=True)
def available_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provide deterministic disk headroom independent of the test workstation."""
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: usage._replace(total=200 * 1024**3, free=150 * 1024**3),
    )


def test_stage_publication_and_reuse(directories: tuple[Path, Path]) -> None:
    """Complete data is immutable and does not carry a run-specific marker."""
    source, output = directories
    with DatasetWriter(output, source) as writer:
        writer.write_bytes("images/a.png", b"image")
        writer.write_bytes("manifest.json", b"{}")
        assert writer.authorized_bytes == 7
        revision = writer.publish("uhd_iqa/revision")
        assert not list(revision.glob("*owner*"))
    assert (revision / "images/a.png").read_bytes() == b"image"
    with DatasetWriter(output, source) as writer:
        assert writer.reuse_path("uhd_iqa/revision") == revision
        with pytest.raises(ApplicationFailure, match="already exists"):
            writer.publish("uhd_iqa/revision")
    assert (revision / "manifest.json").read_bytes() == b"{}"
    assert list((output / ".staging").iterdir()) == []


def test_root_lock_rejects_second_writer(directories: tuple[Path, Path]) -> None:
    """Concurrent dataset names cannot each spend the same free-space reserve."""
    source, output = directories
    with DatasetWriter(output, source):
        with pytest.raises(ApplicationFailure, match="Another dataset"):
            with DatasetWriter(output, source):
                pytest.fail("The second writer must not acquire the root")
    with DatasetWriter(output, source):
        pass


def test_byte_budget_charges_callbacks_once(directories: tuple[Path, Path]) -> None:
    """Every authorized write spends capacity; final file registration does not."""
    source, output = directories
    with DatasetWriter(output, source, maximum_prepared_bytes=10) as writer:
        writer.reserve_write(6)
        (writer.stage / "image.png").write_bytes(b"123456")
        writer.record_external_write("image.png", 6)
        assert writer.authorized_bytes == 6
        writer.reserve_write(4)
        with pytest.raises(ApplicationFailure, match="reserve"):
            writer.reserve_write(1)
    assert list((output / ".staging").iterdir()) == []


def test_disk_reserve_checked_before_each_write(
    directories: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changing free-space result fails safely even after a successful preflight."""
    source, output = directories
    with DatasetWriter(output, source) as writer:
        writer.check_estimated_space(100)
        assert writer.authorized_bytes == 0
        usage = shutil.disk_usage(output)
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda _: usage._replace(total=100 * 1024**3, free=10 * 1024**3),
        )
        with pytest.raises(ApplicationFailure, match="reserve"):
            writer.write_bytes("record.json", b"{}")
    assert list((output / ".staging").iterdir()) == []


def test_interruption_removes_only_current_stage(
    directories: tuple[Path, Path],
) -> None:
    """Ctrl-C preserves the input and any completed revision."""
    source, output = directories
    original = source / "a.png"
    original.write_bytes(b"source")
    with pytest.raises(KeyboardInterrupt):
        with DatasetWriter(output, source) as writer:
            writer.write_bytes("a.png", b"partial")
            raise KeyboardInterrupt
    assert original.read_bytes() == b"source"
    assert list((output / ".staging").iterdir()) == []


def test_recovers_only_owned_staging(directories: tuple[Path, Path]) -> None:
    """Abandoned marked work is removed, while unmarked directories survive."""
    source, output = directories
    staging = output / ".staging"
    staging.mkdir(parents=True)
    name = "run-" + "a" * 32
    (staging / name).mkdir()
    (staging / name / "partial").write_bytes(b"partial")
    (staging / f"{name}.owner.json").write_text(
        json.dumps({"version": 1, "owner_uid": os.getuid(), "stage": name})
    )
    (staging / "unrelated").mkdir()
    with DatasetWriter(output, source):
        assert not (staging / name).exists()
        assert (staging / "unrelated").is_dir()
    assert (staging / "unrelated").is_dir()


def test_stage_path_and_existing_symlink_rejected(
    directories: tuple[Path, Path],
) -> None:
    """Record and revision names cannot escape their selected output roots."""
    source, output = directories
    with DatasetWriter(output, source) as writer:
        with pytest.raises(ApplicationFailure):
            writer.write_bytes("../escape", b"bad")
        (output / "alias").symlink_to(source, target_is_directory=True)
        with pytest.raises(ApplicationFailure):
            writer.reuse_path("alias")
        with pytest.raises(ApplicationFailure):
            writer.publish("alias/revision")
    assert not (source / "revision").exists()


def test_failed_file_flush_preserves_previous_revision(
    directories: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A storage failure never publishes a partial stage or replaces saved data."""
    source, output = directories
    with DatasetWriter(output, source) as writer:
        writer.write_bytes("manifest.json", b"previous")
        completed = writer.publish("uhd_iqa/previous")
    with DatasetWriter(output, source) as writer:
        with monkeypatch.context() as patch:

            def fail_flush(_descriptor: int) -> None:
                """Inject a flush failure at the active write boundary."""
                raise OSError("private filesystem details")

            patch.setattr(os, "fsync", fail_flush)
            with pytest.raises(ApplicationFailure) as failure:
                writer.write_bytes("manifest.json", b"new")
            assert "private" not in str(failure.value)
    assert (completed / "manifest.json").read_bytes() == b"previous"
    assert list((output / ".staging").iterdir()) == []


def test_cleanup_failure_releases_lock_and_can_recover(
    directories: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed cleanup is reported and the next owner can remove the valid stage."""
    source, output = directories
    with monkeypatch.context() as patch:

        def fail_cleanup(_path: Path) -> None:
            """Inject a cleanup error without changing earlier revisions."""
            raise OSError("private cleanup details")

        patch.setattr(shutil, "rmtree", fail_cleanup)
        with pytest.raises(ApplicationFailure):
            with DatasetWriter(output, source) as writer:
                writer.write_bytes("record.json", b"{}")
    assert writer.stage.is_dir()
    with DatasetWriter(output, source):
        assert not writer.stage.exists()


def test_partial_owner_setup_does_not_block_future_imports(
    directories: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted marker write never leaves a corrupt published owner marker."""
    source, output = directories
    with monkeypatch.context() as patch:

        def fail_flush(_descriptor: int) -> None:
            """Inject a flush failure at the active write boundary."""
            raise OSError("disk full")

        patch.setattr(os, "fsync", fail_flush)
        with pytest.raises(ApplicationFailure):
            with DatasetWriter(output, source):
                pytest.fail("The incomplete marker must prevent opening the stage")
    staging = output / ".staging"
    assert list(staging.iterdir()) == []
    pending = staging / ("run-" + "b" * 32 + ".owner.json.pending")
    pending.write_bytes(b'{"incomplete')
    with DatasetWriter(output, source):
        assert not pending.exists()
