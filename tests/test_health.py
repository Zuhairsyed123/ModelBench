"""Integration and unit tests for ModelBench health and root endpoints."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 and expected health metadata schema."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["app_name"] == "ModelBench"
    assert data["version"] == "0.1.0"
    assert data["status"] == "healthy"
    assert "environment" in data
    assert "timestamp" in data


def test_root_endpoint():
    """Verify GET / returns 200 and platform information with links."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["app_name"] == "ModelBench"
    assert data["mode"] == "deterministic_mock"
    assert data["documentation"] == "/docs"
    assert data["health_check"] == "/health"
    assert "endpoints" in data
    assert "benchmark" in data["endpoints"]
