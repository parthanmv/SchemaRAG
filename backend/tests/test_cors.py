"""Regression tests: CORS must accept the local Vite dev server origins.

Root cause fixed: the allowlist pinned Vite's default port 5173 exactly, so
when Vite auto-incremented to 5174 (5173 already occupied) every preflight
was rejected. The middleware now also matches loopback origins on any port
via ``allow_origin_regex`` - never a wildcard, never disabled.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _preflight(client: TestClient, origin: str) -> object:
    return client.options(
        "/api/generate-sql",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",   # documented default (preserved)
        "http://127.0.0.1:5173",   # preserved explicit origin
        "http://localhost:5174",   # Vite auto-increment case (the bug)
        "http://localhost:5199",   # any other auto-incremented dev port
    ],
)
def test_preflight_allowed_for_loopback_dev_origins(client: TestClient, origin):
    resp = _preflight(client, origin)
    assert resp.status_code == 200, resp.text
    assert resp.headers.get("access-control-allow-origin") == origin
    assert "POST" in resp.headers.get("access-control-allow-methods", "")


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example.com",
        "http://schemarag.example.com:8000",
        "null",
    ],
)
def test_preflight_rejected_for_non_loopback_origins(client: TestClient, origin):
    resp = _preflight(client, origin)
    # No ACAO header means the browser blocks the response.
    assert resp.headers.get("access-control-allow-origin") is None


def test_actual_request_gets_acao_header(client: TestClient):
    resp = client.get("/health", headers={"Origin": "http://localhost:5174"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5174"


def test_request_without_origin_unaffected(client: TestClient):
    """Same-service callers (curl, tests) see no CORS headers at all."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers
