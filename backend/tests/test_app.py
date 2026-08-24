"""Phase 1 application tests: startup and /health endpoint."""

from fastapi.testclient import TestClient


def test_app_starts_and_serves_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "SchemaRAG API"
    assert body["health"] == "/health"


def test_health_reports_healthy_with_database(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["database"] == "connected"


def test_openapi_schema_is_generated(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
