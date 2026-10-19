"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from docinsight import __version__


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
