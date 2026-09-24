"""Check exported response contracts and the small headless command surface."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend_service.application import create_application
from backend_service.command_line import main
from backend_service.export_contracts import export_contracts
from schemas.capabilities import Capabilities
from schemas.configuration import ConfigurationProfile
from schemas.jobs import JobEvent


def test_committed_contracts_match_and_detect_drift(tmp_path: Path) -> None:
    """Fail if committed API or entity schemas stop matching Python records."""
    assert export_contracts(Path("contracts"), check=True)
    assert export_contracts(tmp_path)
    assert export_contracts(tmp_path, check=True)
    (tmp_path / "openapi.json").write_text("{}")
    assert not export_contracts(tmp_path, check=True)


def test_serialized_response_contracts_require_default_fields() -> None:
    """Require all promised fields when validating server responses."""
    for name in ("ConfigurationProfile", "Capabilities", "HealthStatus", "ModelList"):
        document = json.loads(Path(f"contracts/entities/{name}.json").read_text())
        assert document["additionalProperties"] is False
        assert set(document["required"]) == set(document["properties"])


def test_model_instances_and_boolean_literals_are_strict() -> None:
    """Reject invalid outgoing instances and integer feature flags."""
    invalid = ConfigurationProfile()
    object.__setattr__(invalid, "checkpoint_interval_seconds", "secret")
    with pytest.raises(ValidationError):
        ConfigurationProfile.model_validate(invalid)
    with pytest.raises(ValidationError):
        Capabilities.model_validate({"encoding_available": 0})
    with pytest.raises(ValidationError):
        JobEvent.model_validate({"password": "secret", "plaintext": "secret"})


def test_command_inspection_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Inspect saved settings and validate files without saving new settings."""
    monkeypatch.setenv("STEGOLAB_DATA_DIRECTORY", str(tmp_path / "data"))
    assert main(["inspect_configuration"]) == 0
    assert json.loads(capsys.readouterr().out)["checkpoint_interval_seconds"] == 300
    configuration_file = tmp_path / "configuration.json"
    configuration_file.write_text('{"checkpoint_interval_seconds":600}')
    assert main(["validate_configuration", str(configuration_file)]) == 0
    assert json.loads(capsys.readouterr().out)["checkpoint_interval_seconds"] == 600
    assert main(["inspect_configuration"]) == 0
    assert json.loads(capsys.readouterr().out)["checkpoint_interval_seconds"] == 300
    configuration_file.write_text('{"password":"private-sentinel"}')
    assert main(["validate_configuration", str(configuration_file)]) == 1
    assert "private-sentinel" not in capsys.readouterr().err


def test_planned_commands_fail_clearly(capsys: pytest.CaptureFixture[str]) -> None:
    """Return a failure code instead of pretending an unavailable command ran."""
    assert main(["encode"]) == 2
    assert "not available" in capsys.readouterr().err


def test_unexpected_failures_do_not_escape_to_server_logs(tmp_path: Path) -> None:
    """Handle unexpected errors inside the server traceback boundary."""
    application = create_application(tmp_path / "data", tmp_path / "logs")

    @application.get("/unexpected-failure")
    def unexpected_failure() -> None:
        """Raise a private sentinel to test the outer exception boundary."""
        raise RuntimeError("secret-unexpected-exception")

    # The default client rethrows exceptions that would escape into uvicorn logs.
    with TestClient(application) as client:
        response = client.get("/unexpected-failure")
        assert response.status_code == 500
        assert "secret-unexpected-exception" not in response.text
    logs = "".join(path.read_text() for path in (tmp_path / "logs").rglob("*.jsonl"))
    assert "secret-unexpected-exception" not in logs
    assert "application_request_failed" in logs
