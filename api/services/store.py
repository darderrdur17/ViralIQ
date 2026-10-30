"""Prediction and observation persistence.

Predictions are stored so that their eventual outcomes can be measured. A model
that cannot be scored against reality later is not a forecasting system.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import duckdb

from src.common.config import SETTINGS
from src.common.logging import get_logger
from src.ingestion import duckdb_engine as db

log = get_logger("store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     VARCHAR PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL,
    task              VARCHAR NOT NULL,
    model_version     VARCHAR NOT NULL,
    viral_probability DOUBLE,
    virality_tier     VARCHAR,
    predicted_percentile DOUBLE,
    payload           JSON,
    response          JSON
);

CREATE TABLE IF NOT EXISTS video_observations (
    observation_id    VARCHAR PRIMARY KEY,
    video_id          UBIGINT,
    observed_at       TIMESTAMPTZ NOT NULL,
    observation_age_hours DOUBLE,
    views             UBIGINT,
    likes             UBIGINT,
    comments          UBIGINT,
    shares            UBIGINT,
    saves             UBIGINT,
    source            VARCHAR,
    raw               JSON
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id            VARCHAR PRIMARY KEY,
    started_at        TIMESTAMPTZ NOT NULL,
    finished_at       TIMESTAMPTZ,
    source            VARCHAR,
    rows_ingested     BIGINT,
    status            VARCHAR,
    detail            JSON
);
"""


_CONN: duckdb.DuckDBPyConnection | None = None
_LOCK = threading.RLock()
_SCHEMA_READY = False
_DB_PATH: str | None = None


def configure(db_path: str | None) -> None:
    """Point the store at a specific database file, or back to the default.

    Tests use this to avoid competing for the lock on the live application database.
    DuckDB holds an OS-level lock per file, so a running server and a test process
    cannot share one.
    """
    global _CONN, _SCHEMA_READY, _DB_PATH
    with _LOCK:
        if _CONN is not None:
            try:
                _CONN.close()
            except Exception:  # pragma: no cover - defensive
                pass
        _CONN = None
        _SCHEMA_READY = False
        _DB_PATH = db_path


@contextlib.contextmanager
def _session() -> Iterator[duckdb.DuckDBPyConnection]:
    """Serialised access to the application database.

    DuckDB permits only one handle per database file within a process, so opening a
    fresh connection per request raises ``Unique file handle conflict`` as soon as two
    requests overlap — which is exactly what a prediction endpoint sees. A single
    shared connection, guarded by a lock, is the correct shape here: writes are tiny
    and correctness matters more than parallel write throughput.

    Note that the lock is per *file*, not per process: this store therefore assumes a
    single application process. Running multiple uvicorn workers against one database
    file will fail at startup, by design of DuckDB.
    """
    global _CONN, _SCHEMA_READY
    with _LOCK:
        if _CONN is None:
            target = (
                _DB_PATH
                or os.environ.get("VIRALIQ_DB_PATH")
                or str(SETTINGS.paths.duckdb_file)
            )
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            _CONN = db.connect(target)
        if not _SCHEMA_READY:
            for statement in filter(None, (s.strip() for s in SCHEMA.split(";"))):
                _CONN.execute(statement)
            _SCHEMA_READY = True
        yield _CONN


def init_db() -> None:
    with _session():
        pass


def save_prediction(task: str, model_version: str, payload: dict, response: dict) -> str:
    prediction_id = str(uuid.uuid4())
    with _session() as con:
        con.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                prediction_id,
                datetime.now(timezone.utc),
                task,
                model_version,
                float(response.get("viral_probability") or 0.0),
                response.get("virality_tier"),
                response.get("predicted_percentile"),
                json.dumps(payload, default=str),
                json.dumps(response, default=str),
            ],
        )
    return prediction_id


def list_predictions(limit: int = 50, task: str | None = None) -> list[dict]:
    with _session() as con:
        sql = (
            "SELECT prediction_id, created_at, task, model_version, viral_probability, "
            "virality_tier, predicted_percentile, payload, response "
            "FROM predictions"
        )
        params: list[Any] = []
        if task:
            sql += " WHERE task = ?"
            params.append(task)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = con.execute(sql, params).fetchall()
        cols = [d[0] for d in con.description]

    out = []
    for row in rows:
        record = dict(zip(cols, row, strict=True))
        for key in ("payload", "response"):
            if isinstance(record.get(key), str):
                try:
                    record[key] = json.loads(record[key])
                except json.JSONDecodeError:
                    pass
        out.append(record)
    return out


def prediction_stats() -> dict:
    with _session() as con:
        total = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
        observations = con.execute("SELECT count(*) FROM video_observations").fetchone()[0]
        if total:
            avg = con.execute("SELECT avg(viral_probability) FROM predictions").fetchone()[0]
            latest = con.execute("SELECT max(created_at) FROM predictions").fetchone()[0]
        else:
            avg, latest = None, None
    return {
        "total_predictions": int(total),
        "mean_viral_probability": round(float(avg), 4) if avg is not None else None,
        "latest_prediction_at": latest.isoformat() if latest else None,
        "observations_recorded": int(observations),
    }


def add_observation(video_id: int, observed_at: datetime, counts: dict, source: str,
                    observation_age_hours: float | None = None) -> str:
    observation_id = str(uuid.uuid4())
    with _session() as con:
        con.execute(
            "INSERT INTO video_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                observation_id,
                int(video_id),
                observed_at,
                observation_age_hours,
                counts.get("views"),
                counts.get("likes"),
                counts.get("comments"),
                counts.get("shares"),
                counts.get("saves"),
                source,
                json.dumps(counts, default=str),
            ],
        )
    return observation_id


def observation_count() -> int:
    with _session() as con:
        return int(con.execute("SELECT count(*) FROM video_observations").fetchone()[0])
