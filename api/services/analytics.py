"""Analytics over the ingested sample.

Every number returned here is a SQL aggregate over files on disk. Nothing is
pre-computed, mocked, or hard-coded — if the sample is empty the endpoint says so
rather than inventing a chart.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.config import SETTINGS
from src.common.logging import get_logger
from src.ingestion import duckdb_engine as db

log = get_logger("analytics")


def _is_complete_parquet(path) -> bool:
    """A file still being written has no trailing PAR1 magic.

    Ingestions run while the API is up, so discovery must tolerate partial files
    rather than crashing every analytics query.
    """
    try:
        if path.stat().st_size < 12:
            return False
        with path.open("rb") as handle:
            head = handle.read(4)
            handle.seek(-4, 2)
            tail = handle.read(4)
        return head == b"PAR1" and tail == b"PAR1"
    except OSError:
        return False


def sample_files() -> list[str]:
    files = sorted(SETTINGS.paths.data_samples.glob("*.parquet"))
    return [str(f) for f in files if _is_complete_parquet(f)]


def sample_weights() -> dict[str, float]:
    """Per-file inverse sampling probability.

    Sample files may be built at different fractions (a 1% month alongside a full
    month). Raw counts are then not comparable across files: an "acceleration"
    computed from raw counts would measure the difference in sampling rate rather
    than any real change in behaviour. Hash sampling is unbiased, so multiplying a
    count by 1/fraction recovers an estimate of the population count.
    """
    weights: dict[str, float] = {}
    for meta_path in SETTINGS.paths.data_samples.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        fraction = meta.get("fraction") or 1.0
        weight = 1.0 / fraction if fraction > 0 else 1.0
        for path in meta.get("paths") or []:
            weights[Path(path).name] = weight
    return weights


def _weighted_relation(files: list[str]) -> tuple[str, bool]:
    """Relation exposing a `weight` column, or (relation, False) if weights are uniform."""
    weights = sample_weights()
    per_file = {Path(f).name: weights.get(Path(f).name, 1.0) for f in files}
    uniform = len({round(w, 6) for w in per_file.values()}) == 1

    joined = ", ".join(f"'{f}'" for f in files)
    base = f"read_parquet([{joined}], filename=true)"
    if uniform:
        return f"(SELECT *, 1.0 AS weight FROM {base})", True

    values = ", ".join(f"('{name}', {w})" for name, w in per_file.items())
    rel = (
        f"(SELECT t.*, w.weight FROM {base} t "
        f"JOIN (VALUES {values}) AS w(fname, weight) "
        f"ON ends_with(t.filename, w.fname))"
    )
    return rel, False


def _relation(files: list[str]) -> str:
    return db.remote_scan(files)


def is_ready() -> bool:
    return bool(sample_files())


def overview() -> dict:
    files = sample_files()
    if not files:
        return {"available": False, "reason": "no_sample_data",
                "message": "No sample data ingested. Run scripts/build_sample.py first."}

    rel, uniform = _weighted_relation(files)
    with db.session() as con:
        row = con.execute(
            f"""
            SELECT count(*)                              AS rows_sampled,
                   round(sum(weight), 0)                 AS rows_estimated,
                   min(posted_at)                        AS min_posted_at,
                   max(posted_at)                        AS max_posted_at,
                   count(DISTINCT country)               AS countries,
                   count(DISTINCT language)              AS languages,
                   count(DISTINCT sound_id)              AS sounds,
                   round(median(views), 0)               AS median_views,
                   round(avg(views), 0)                  AS mean_views,
                   round(quantile_cont(views, 0.99), 0)  AS p99_views,
                   round(100.0 * avg(CASE WHEN views = 0 THEN 1 ELSE 0 END), 2) AS pct_zero_views
            FROM {rel}
            """
        ).fetchone()
        cols = [d[0] for d in con.description]
        base = dict(zip(cols, row, strict=True))

        monthly = con.execute(
            f"""
            SELECT strftime(posted_at, '%Y-%m') AS month,
                   count(*) AS videos_sampled,
                   round(sum(weight), 0) AS videos_estimated,
                   round(quantile_cont(views, 0.99), 0) AS p99_views,
                   round(median(views), 0) AS median_views
            FROM {rel}
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()

    for key in ("min_posted_at", "max_posted_at"):
        base[key] = base[key].isoformat() if base[key] else None

    base["available"] = True
    base["rows"] = int(base.pop("rows_sampled"))
    base["estimated_population_rows"] = int(base.pop("rows_estimated") or base["rows"])
    base["sampling_is_uniform"] = uniform
    base["sample_files"] = [f.split("/")[-1] for f in files]
    base["monthly"] = [
        {
            "month": m,
            "videos": int(vs),
            "videos_estimated": int(ve or vs),
            "p99_views": float(p),
            "median_views": float(md),
        }
        for m, vs, ve, p, md in monthly
    ]
    return base


def trends(limit: int = 15) -> dict:
    files = sample_files()
    if not files:
        return {"available": False, "reason": "no_sample_data"}

    rel, uniform = _weighted_relation(files)
    with db.session() as con:
        hashtags = con.execute(
            f"""
            WITH h AS (
                SELECT unnest(hashtags) AS tag, posted_at, weight
                FROM {rel}
                WHERE hashtags IS NOT NULL
            ),
            bounds AS (SELECT max(posted_at) AS mx FROM h),
            agg AS (
                SELECT tag,
                       sum(weight) AS uses,
                       COALESCE(sum(weight) FILTER (
                           WHERE posted_at > (SELECT mx FROM bounds) - INTERVAL 7 DAY
                       ), 0) AS uses_last_7d,
                       COALESCE(sum(weight) FILTER (
                           WHERE posted_at <= (SELECT mx FROM bounds) - INTERVAL 7 DAY
                       ), 0) AS uses_prior
                FROM h GROUP BY 1
            )
            SELECT tag, uses, uses_last_7d, uses_prior,
                   CASE WHEN uses_prior >= 50
                        THEN round(uses_last_7d / uses_prior, 3)
                        ELSE NULL END AS acceleration,
                   CASE WHEN uses_prior >= 50 THEN 'accelerating' ELSE 'emerging' END AS pattern
            FROM agg WHERE uses >= 50
            ORDER BY (uses_prior < 50) ASC, acceleration DESC NULLS LAST, uses_last_7d DESC
            LIMIT {int(limit)}
            """
        ).fetchall()

        sounds = con.execute(
            f"""
            WITH s AS (
                SELECT sound_id, posted_at, views, weight FROM {rel} WHERE sound_id IS NOT NULL
            ),
            bounds AS (SELECT max(posted_at) AS mx FROM s),
            agg AS (
                SELECT sound_id,
                       sum(weight) AS videos,
                       COALESCE(sum(weight) FILTER (
                           WHERE posted_at > (SELECT mx FROM bounds) - INTERVAL 7 DAY
                       ), 0) AS videos_last_7d,
                       COALESCE(sum(weight) FILTER (
                           WHERE posted_at <= (SELECT mx FROM bounds) - INTERVAL 7 DAY
                       ), 0) AS videos_prior,
                       round(median(views), 0) AS median_views
                FROM s GROUP BY 1
            )
            SELECT sound_id, videos, videos_last_7d, videos_prior, median_views,
                   CASE WHEN videos_prior >= 20
                        THEN round(videos_last_7d / videos_prior, 3)
                        ELSE NULL END AS acceleration,
                   CASE WHEN videos_prior >= 20 THEN 'accelerating' ELSE 'emerging' END AS pattern
            FROM agg WHERE videos >= 20
            ORDER BY (videos_prior < 20) ASC, acceleration DESC NULLS LAST, videos_last_7d DESC
            LIMIT {int(limit)}
            """
        ).fetchall()

        window = con.execute(f"SELECT min(posted_at), max(posted_at) FROM {rel}").fetchone()

    note = (
        "Acceleration compares the final 7 days of the observed window against all earlier "
        "days in that window. Counts are weighted by the inverse sampling fraction, so a "
        "1%-sampled month and a fully-sampled month are comparable. These are historical "
        "observations, not a live trend feed. A tag marked 'emerging' had no meaningful "
        "prior usage inside the window, so no ratio is reported for it — a first appearance "
        "is not an acceleration."
    )
    if not uniform:
        note += (
            " Sample files were built at different fractions; weights correct for this, but "
            "estimates from the sparsely sampled file carry more sampling error."
        )

    return {
        "available": True,
        "window": {
            "start": window[0].isoformat() if window[0] else None,
            "end": window[1].isoformat() if window[1] else None,
            "acceleration_note": note,
            "counts_are_weighted": not uniform,
        },
        "accelerating_hashtags": [
            {"hashtag": r[0], "uses": round(float(r[1])), "uses_last_7d": round(float(r[2])),
             "uses_prior": round(float(r[3])),
             "acceleration": float(r[4]) if r[4] is not None else None,
             "pattern": r[5]}
            for r in hashtags
        ],
        "accelerating_sounds": [
            {"sound_id": int(r[0]), "videos": round(float(r[1])), "videos_last_7d": round(float(r[2])),
             "videos_prior": round(float(r[3])),
             "median_views": float(r[4]) if r[4] is not None else None,
             "acceleration": float(r[5]) if r[5] is not None else None,
             "pattern": r[6]}
            for r in sounds
        ],
    }


def content_analytics() -> dict:
    files = sample_files()
    if not files:
        return {"available": False, "reason": "no_sample_data"}

    rel, _ = _weighted_relation(files)
    with db.session() as con:
        duration = con.execute(
            f"""
            SELECT CASE
                     WHEN duration_ms < 8000 THEN '0-8s'
                     WHEN duration_ms < 15000 THEN '8-15s'
                     WHEN duration_ms < 30000 THEN '15-30s'
                     WHEN duration_ms < 60000 THEN '30-60s'
                     WHEN duration_ms < 180000 THEN '1-3min'
                     ELSE '3min+'
                   END AS bucket,
                   round(sum(weight)) AS videos,
                   round(median(views), 0) AS median_views,
                   round(quantile_cont(views, 0.99), 0) AS p99_views
            FROM {rel} WHERE duration_ms > 0
            GROUP BY 1 ORDER BY min(duration_ms)
            """
        ).fetchall()

        hours = con.execute(
            f"""
            SELECT extract(hour FROM posted_at) AS hour_utc,
                   round(sum(weight)) AS videos,
                   round(median(views), 0) AS median_views
            FROM {rel} GROUP BY 1 ORDER BY 1
            """
        ).fetchall()

        countries = con.execute(
            f"""
            SELECT country, round(sum(weight)) AS videos, round(median(views), 0) AS median_views,
                   round(quantile_cont(views, 0.99), 0) AS p99_views
            FROM {rel} GROUP BY 1 HAVING sum(weight) >= 100
            ORDER BY videos DESC LIMIT 15
            """
        ).fetchall()

    return {
        "available": True,
        "duration_buckets": [
            {"bucket": r[0], "videos": r[1], "median_views": float(r[2] or 0),
             "p99_views": float(r[3] or 0)} for r in duration
        ],
        "hour_utc": [
            {"hour": int(r[0]), "videos": r[1], "median_views": float(r[2] or 0)} for r in hours
        ],
        "countries": [
            {"country": r[0], "videos": r[1], "median_views": float(r[2] or 0),
             "p99_views": float(r[3] or 0)} for r in countries
        ],
    }


def virality_prevalence() -> dict:
    """Observed share of videos above the p99 threshold, by month."""
    files = sample_files()
    if not files:
        return {"available": False}
    rel = _relation(files)
    with db.session() as con:
        rows = con.execute(
            f"""
            WITH t AS (
                SELECT strftime(posted_at, '%Y-%m') AS month, views FROM {rel}
            ),
            thr AS (SELECT quantile_cont(views, 0.99) AS t FROM t)
            SELECT month, count(*) AS videos,
                   round(100.0 * avg(CASE WHEN views >= (SELECT t FROM thr) THEN 1 ELSE 0 END), 3) AS pct_above_p99
            FROM t GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
    return {
        "available": True,
        "definition": "Share of videos at or above the global p99 view count of the ingested sample.",
        "by_month": [{"month": r[0], "videos": r[1], "pct_above_p99": float(r[2])} for r in rows],
    }
