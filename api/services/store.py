"""Prediction and observation persistence.

Predictions are stored so that their eventual outcomes can be measured. A model
that cannot be scored against reality later is not a forecasting system.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

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


def init_db() -> None:
    con = db.app_db()
    try:
        for statement in filter(None, (s.strip() for s in SCHEMA.split(";"))):
            con.execute(statement)
    finally:
        con.close()


def save_prediction(task: str, model_version: str, payload: dict, response: dict) -> str:
    init_db()
    prediction_id = str(uuid.uuid4())
    con = db.app_db()
    try:
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
    finally:
        con.close()
    return prediction_id


def list_predictions(limit: int = 50, task: str | None = None) -> list[dict]:
    init_db()
    con = db.app_db()
    try:
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
    finally:
        con.close()

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
    init_db()
    con = db.app_db()
    try:
        total = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
        observations = con.execute("SELECT count(*) FROM video_observations").fetchone()[0]
        if total:
            avg = con.execute("SELECT avg(viral_probability) FROM predictions").fetchone()[0]
            latest = con.execute("SELECT max(created_at) FROM predictions").fetchone()[0]
        else:
            avg, latest = None, None
    finally:
        con.close()
    return {
        "total_predictions": int(total),
        "mean_viral_probability": round(float(avg), 4) if avg is not None else None,
        "latest_prediction_at": latest.isoformat() if latest else None,
        "observations_recorded": int(observations),
    }


def add_observation(video_id: int, observed_at: datetime, counts: dict, source: str,
                    observation_age_hours: float | None = None) -> str:
    init_db()
    observation_id = str(uuid.uuid4())
    con = db.app_db()
    try:
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
    finally:
        con.close()
    return observation_id


def observation_count() -> int:
    init_db()
    con = db.app_db()
    try:
        return int(con.execute("SELECT count(*) FROM video_observations").fetchone()[0])
    finally:
        con.close()
