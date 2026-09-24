"""CLI demos expose safe metadata and keep model commands unavailable."""

import json
from pathlib import Path

import pytest

from backend_service.command_line import main
from backend_service.failures import ApplicationFailure
from backend_service.protocol_verification import verify_protocol


def test_public_demo(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The demo succeeds using packaged vectors and creates no durable state."""
    monkeypatch.chdir(tmp_path)
    assert main(["verify_protocol"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["fixture_count"] == 4
    assert result["status"] == "passed"
    assert "correctable_damage" in result["checks_passed"]
    assert "password" not in result
    assert captured.err == ""
    assert list(tmp_path.iterdir()) == []


def test_verification_rejects_secret_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unsupported arguments cannot become a password-taking diagnostic mode."""
    with pytest.raises(SystemExit) as caught:
        main(["verify_protocol", "--password", "PRIVATE_SENTINEL"])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "PRIVATE_SENTINEL" not in captured.out + captured.err


def test_broken_fixture_is_a_safe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return failure if static bytes no longer match, without printing them."""

    def mismatch(*arguments: object, **keywords: object) -> bytes:
        """Inject a failing comparison into public verification."""
        return b"PRIVATE_SENTINEL"

    monkeypatch.setattr("backend_service.protocol_verification.protect_frame", mismatch)
    with pytest.raises(ApplicationFailure) as caught:
        verify_protocol()
    assert caught.value.code == "protocol_verification_failed"
    assert "PRIVATE_SENTINEL" not in str(caught.value)


@pytest.mark.parametrize("command", ["encode", "decode"])
def test_model_commands_stay_unavailable(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Protocol tools do not claim to implement actual steganography."""
    assert main([command]) == 2
    assert "not available" in capsys.readouterr().err
