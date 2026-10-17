"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from docinsight import __version__
from docinsight.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
