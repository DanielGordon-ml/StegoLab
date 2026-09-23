"""Small isolated API fixtures that never use production data directories."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend_service.application import create_application


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Start an API using a new database and log directory for each test."""
    application = create_application(tmp_path / "data", tmp_path / "logs")
    with TestClient(application, raise_server_exceptions=False) as connection:
        yield connection
