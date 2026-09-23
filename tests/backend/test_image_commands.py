"""Image CLI commands report safe metadata and preserve existing output."""

import json
from pathlib import Path

import pytest
from PIL import Image

from backend_service.command_line import main
from backend_service.image_preparation import read_prepared_png


def test_image_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Inspect, prepare, and reopen pixels through the user-facing CLI."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "private_source.png"
    output = tmp_path / "prepared.png"
    with Image.new("RGBA", (513, 517), (24, 42, 60, 127)) as image:
        image.save(source)
        expected = image.tobytes()
    assert main(["inspect_image", str(source)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["prepared_width"] == 513
    assert inspected["color_policy"] == "assumed_srgb"
    assert main(["prepare_image", str(source), str(output)]) == 0
    captured = capsys.readouterr()
    assert str(source) not in captured.out + captured.err
    assert json.loads(captured.out)["pixel_policy"] == "prepare_srgb"
    prepared = read_prepared_png(output)
    assert prepared.image.tobytes() == expected
    original_output = output.read_bytes()
    assert main(["prepare_image", str(source), str(output)]) == 1
    assert output.read_bytes() == original_output
    assert not (tmp_path / ".runtime").exists()


def test_image_command_failure_is_private(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing or invalid file produces a fixed error without a traceback."""
    source = tmp_path / "PRIVATE_PATH_SENTINEL.png"
    source.write_bytes(b"PRIVATE_CONTENT_SENTINEL")
    assert main(["inspect_image", str(source)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "PRIVATE_" not in captured.err
    assert "Traceback" not in captured.err
