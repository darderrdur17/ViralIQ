"""API contract tests.

These assert structure and honesty guarantees, not specific metric values, so they
stay valid across retrainings.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_health_reports_data_mode(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # The service must always declare which data mode it is in.
    assert body["data_mode"] in {"live", "recently_refreshed", "historical", "unavailable"}


def test_meta_declares_blocked_task(client):
    body = client.get("/api/v1/meta").json()
    tasks = {t["task"]: t for t in body["prediction_tasks"]}
    assert tasks["prepublication"]["available"] is True
    # Early traction must be declared unavailable rather than silently absent.
    assert tasks["early_traction"]["available"] is False


def test_data_status_includes_freshness_and_quality(client):
    body = client.get("/api/v1/data/status").json()
    for key in ("mode", "publication_delay_days", "data_quality", "model"):
        assert key in body
    assert isinstance(body["data_quality"]["fixed_age_label_supported"], (bool, type(None)))


def test_overview_is_measured_or_explicitly_unavailable(client):
    body = client.get("/api/v1/analytics/overview").json()
    if body.get("available"):
        assert body["rows"] > 0
        assert body["max_posted_at"] >= body["min_posted_at"]
    else:
        assert body["reason"] == "no_sample_data"


def test_prepublish_rejects_engagement_fields(client):
    """Engagement must be refused outright, not silently dropped.

    Silently ignoring ``views`` would let a caller believe they supplied engagement
    data and still receive a prediction, which is exactly the confusion the
    pre-publication contract exists to prevent.
    """
    r = client.post(
        "/api/v1/predictions/prepublish",
        json={"caption": "hello", "views": 1_000_000},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any("views" in str(item.get("loc", "")) for item in detail)


def test_prepublish_flags_incomplete_input(client):
    """An empty request is allowed but must declare what was defaulted."""
    r = client.post("/api/v1/predictions/prepublish", json={})
    if r.status_code == 503:
        pytest.skip("No trained model available in this environment")
    assert r.status_code == 200
    body = r.json()
    assert body["input_warnings"], "an empty request must report defaulted fields"
    assert any("caption" in w for w in body["input_warnings"])


def test_label_definition_names_the_age_bucket(client):
    """The cohort includes observation age; the description must not omit it."""
    r = client.post(
        "/api/v1/predictions/prepublish",
        json={"caption": "x", "duration_ms": 9000, "country": "US", "language": "en"},
    )
    if r.status_code == 503:
        pytest.skip("No trained model available in this environment")
    label = r.json()["label_definition"].lower()
    assert "observation age" in label, f"label definition is stale: {label!r}"


def test_prepublish_returns_calibrated_prediction(client):
    r = client.post(
        "/api/v1/predictions/prepublish",
        json={
            "caption": "three ingredient pasta hack",
            "hashtags": ["foodtok", "recipe"],
            "duration_ms": 21_000,
            "country": "US",
            "language": "en",
        },
    )
    if r.status_code == 503:
        pytest.skip("No trained model available in this environment")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["viral_probability"] <= 1.0
    assert body["prediction_id"]
    assert body["limitations"], "predictions must carry their limitations"
    assert body["model"]["version"]


def test_early_traction_refuses_without_observations(client):
    r = client.post(
        "/api/v1/predictions/early-traction",
        json={"video_id": 1, "views": 5000, "observation_age_hours": 6},
    )
    assert r.status_code in (409, 501)
    detail = r.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["error"] in {"insufficient_observations", "model_not_trained"}


def test_prediction_history_is_persisted(client):
    client.post(
        "/api/v1/predictions/prepublish",
        json={"caption": "history check", "duration_ms": 9000, "country": "US"},
    )
    body = client.get("/api/v1/predictions/history?limit=5").json()
    assert "items" in body and "stats" in body
    assert isinstance(body["count"], int)


def test_models_endpoint_degrades_cleanly(client):
    r = client.get("/api/v1/models")
    assert r.status_code in (200, 503)
    if r.status_code == 503:
        assert "registry" in r.json()["detail"].lower()


def test_import_parquet_end_to_end(client):
    """Regression: Parquet import used to fail for every file."""
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    import polars as pl

    base = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rows = {
        "video_id": [1, 2],
        "posted_at": [base, base.replace(hour=1)],
        "country": ["US", "ID"],
        "language": ["en", "id"],
        "type": ["video", "video"],
        "duration_ms": [10_000, 20_000],
        "caption": ["a", "b"],
        "hashtags": [["x"], ["y"]],
        "sound_id": [1, 2],
        "views": [100, 200],
        "likes": [1, 2],
        "comments": [0, 0],
        "shares": [0, 0],
        "saves": [0, 0],
        "stats_updated_at": [base.replace(hour=5), base.replace(hour=6)],
    }
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "api_import_probe.parquet"
        pl.DataFrame(rows).write_parquet(path)
        r = client.post("/api/v1/data/import", json={"path": str(path), "source_label": "pytest"})
        assert r.status_code == 200, r.json()
        body = r.json()
        assert body["rows"] == 2
        assert body["stored_at"].endswith(".parquet")
        # Clean up so repeated runs stay idempotent.
        Path(body["stored_at"]).unlink(missing_ok=True)


def test_import_missing_file_is_404(client):
    r = client.post("/api/v1/data/import", json={"path": "/tmp/definitely-not-here.parquet"})
    assert r.status_code == 404


def test_import_bad_schema_is_422(client):
    import tempfile
    from pathlib import Path

    import polars as pl

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bad_schema.parquet"
        pl.DataFrame({"foo": [1]}).write_parquet(path)
        r = client.post("/api/v1/data/import", json={"path": str(path), "source_label": "pytest"})
        assert r.status_code == 422
        assert "missing required columns" in str(r.json()["detail"])


def test_unknown_route_is_404(client):
    assert client.get("/api/v1/nope").status_code == 404


def test_post_route_rejects_get(client):
    assert client.get("/api/v1/predictions/prepublish").status_code == 405
