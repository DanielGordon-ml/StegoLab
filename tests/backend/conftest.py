"""Small isolated API fixtures that never use production data directories."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from backend_service.application import create_application
from backend_service.model_fixture_channel import (
    FIXTURE_PACKAGE_NAME,
    build_fixture_package,
)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Start an API using a new database and log directory for each test."""
    application = create_application(tmp_path / "data", tmp_path / "logs")
    with TestClient(application, raise_server_exceptions=False) as connection:
        yield connection


@pytest.fixture(scope="session")
def fixture_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the labelled fixture pair once into an isolated workspace root."""
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        root = tmp_path_factory.mktemp("fixture-workspace")
        # The builder creates the missing models folder, as CI relies on it.
        build_fixture_package(root / "models" / FIXTURE_PACKAGE_NAME)
        return root
    finally:
        torch.set_num_threads(original_threads)
