"""Tests for the HEAD middleware and the schema endpoint contract."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.middleware import HeadAsGetMiddleware


@pytest.fixture()
def head_client() -> TestClient:
    app = FastAPI()
    app.add_middleware(HeadAsGetMiddleware)

    @app.get("/thing")
    def thing() -> dict:
        return {"hello": "world", "value": 42}

    @app.get("/boom")
    def boom() -> dict:
        raise RuntimeError("expected failure")

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_head_returns_get_status(head_client):
    """FastAPI's APIRoute omits HEAD; the middleware must supply it."""
    assert head_client.head("/thing").status_code == 200


def test_head_has_no_body_but_reports_length(head_client):
    response = head_client.head("/thing")
    assert response.content == b"", "HEAD responses must not carry a body"
    get_response = head_client.get("/thing")
    assert response.headers.get("content-length") == get_response.headers.get("content-length")


def test_get_is_unaffected(head_client):
    assert head_client.get("/thing").json() == {"hello": "world", "value": 42}


def test_head_propagates_error_status(head_client):
    """HEAD must mirror GET's status, not swallow failures."""
    assert head_client.get("/boom").status_code == 500
    assert head_client.head("/boom").status_code == 500


def test_head_on_missing_route_is_404(head_client):
    assert head_client.head("/nope").status_code == 404


def test_full_service_head_on_api_routes():
    """The real app must answer HEAD on its health endpoint."""
    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.head("/api/v1/health").status_code == 200
        assert client.head("/api/v1/analytics/overview").status_code == 200
        assert client.head("/").status_code in (200, 307)


def test_schema_endpoint_declares_availability():
    """Both deployments answer the same shape; availability differs."""
    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/api/v1/data/schema").json()
        assert body["available"] is True
        assert "columns" in body and "check" in body
