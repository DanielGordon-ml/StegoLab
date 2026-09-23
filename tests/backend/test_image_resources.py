"""Keep image allocation failures safe and preserve existing files."""

import traceback
from pathlib import Path

import pytest
from PIL import Image

from backend_service.failures import ApplicationFailure
from backend_service.image_preparation import prepare_image


@pytest.mark.parametrize("operation", ["decode", "save", "verify"])
@pytest.mark.parametrize("existing_output", [False, True])
def test_memory_failure_preserves_files_and_hides_private_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    existing_output: bool,
) -> None:
    """Fail source allocation or output work without private tracebacks or debris."""
    source = tmp_path / "private-source.png"
    Image.new("RGBA", (512, 512), (19, 41, 78, 120)).save(source)
    source_bytes = source.read_bytes()
    output = tmp_path / "private-output.png"
    previous_output = b"existing-output-must-survive"
    if existing_output:
        output.write_bytes(previous_output)

    def memory_failure(*args: object, **kwargs: object) -> None:
        """Model an allocation failure containing private decoder details."""
        raise MemoryError("private-source.png secret-memory-sentinel")

    if operation == "decode":
        monkeypatch.setattr(Image.Image, "copy", memory_failure)
    elif operation == "save":
        monkeypatch.setattr(Image.Image, "save", memory_failure)
    else:
        monkeypatch.setattr(
            "backend_service.image_output.ImageChops.difference", memory_failure
        )
    with pytest.raises(ApplicationFailure) as failure:
        prepare_image(source, output)
    assert failure.value.code == "image_resources"
    assert failure.value.status_code == 503
    assert "memory" in str(failure.value)
    rendered = "".join(traceback.format_exception(failure.value))
    assert "secret-memory-sentinel" not in rendered
    assert "private-source.png" not in rendered
    assert source.read_bytes() == source_bytes
    if existing_output:
        assert output.read_bytes() == previous_output
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".stegolab-image-*"))
