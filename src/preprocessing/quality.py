"""Data quality and integrity checks.

The single most important integrity property for this project is *observation
age*: the gap between when a video was posted and when its engagement counters
were read. Without a trustworthy observation timestamp there is no way to build a
fixed-age target, and any "7-day" label would be fiction.

This module measures that property rather than assuming it.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import duckdb

from src.common.logging import get_logger

log = get_logger("quality")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "video_id",
    "posted_at",
    "country",
    "language",
    "type",
    "duration_ms",
    "caption",
    "hashtags",
    "sound_id",
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "stats_updated_at",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "image_count",
    "width",
    "height",
    "mentions",
    "on_screen_text",
    "hashtag_ids",
    "is_ad",
    "branded_content",
    "product_id",
    "seller_id",
    "is_shop_video",
    "is_ai_generated",
    "not_recommended_for_fyp",
    "sound_muted",
    "downloads",
    "engagement_rate",
)

OBSERVATION_AGE_HOURS = "(epoch(stats_updated_at) - epoch(posted_at)) / 3600.0"


@dataclasses.dataclass(frozen=True)
class SchemaCheck:
    present: list[str]
    missing_required: list[str]
    missing_optional: list[str]

    @property
    def valid(self) -> bool:
        return not self.missing_required

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "column_count": len(self.present),
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
        }


def validate_schema(columns: list[str]) -> SchemaCheck:
    present = list(columns)
    have = set(present)
    return SchemaCheck(
        present=present,
        missing_required=[c for c in REQUIRED_COLUMNS if c not in have],
        missing_optional=[c for c in OPTIONAL_COLUMNS if c not in have],
    )


def audit_relation(con: duckdb.DuckDBPyConnection, relation_sql: str, label: str) -> dict:
    """Profile one Parquet relation in a single streaming pass.

    Deliberately one aggregate query rather than a CTE re-referenced by many
    scalar subqueries: the latter makes DuckDB materialise the whole relation once
    per subquery, which exhausts memory on a 100M-row partition. Approximate
    quantiles are used because exact quantiles require a full in-memory sort.

    Returns measured facts only — no extrapolation, no estimates.
    """
    sql = f"""
    SELECT
        count(*)                                                              AS row_count,
        min(posted_at)                                                        AS min_posted_at,
        max(posted_at)                                                        AS max_posted_at,
        min(stats_updated_at)                                                 AS min_stats_updated_at,
        max(stats_updated_at)                                                 AS max_stats_updated_at,
        count(DISTINCT country)                                               AS distinct_countries,
        count(DISTINCT language)                                              AS distinct_languages,
        count(DISTINCT sound_id)                                              AS distinct_sounds,
        count(DISTINCT stats_updated_at)                                      AS distinct_stats_timestamps,
        round(avg(views), 2)                                                  AS mean_views,
        round(approx_quantile(views, 0.5), 2)                                 AS median_views,
        round(approx_quantile(views, 0.99), 2)                                AS p99_views,
        round(approx_quantile(views, 0.999), 2)                               AS p999_views,
        round(100.0 * avg(CASE WHEN views = 0 THEN 1 ELSE 0 END), 4)          AS pct_zero_views,
        round(100.0 * avg(CASE WHEN stats_updated_at IS NULL THEN 1 ELSE 0 END), 4)
                                                                              AS pct_null_stats_ts,
        round(100.0 * avg(CASE WHEN {OBSERVATION_AGE_HOURS} < 0 THEN 1 ELSE 0 END), 4)
                                                                              AS pct_negative_observation_age,
        round(100.0 * avg(CASE WHEN {OBSERVATION_AGE_HOURS} > 720 THEN 1 ELSE 0 END), 4)
                                                                              AS pct_age_over_30d,
        round(approx_quantile({OBSERVATION_AGE_HOURS}, 0.5), 2)                AS observation_age_p50_h,
        round(approx_quantile({OBSERVATION_AGE_HOURS}, 0.99), 2)               AS observation_age_p99_h
    FROM {relation_sql}
    """
    row = con.execute(sql).fetchone()
    keys = [d[0] for d in con.description]
    result = dict(zip(keys, row, strict=True))
    result["label"] = label
    result["measured_at"] = datetime.now(timezone.utc).isoformat()
    result["quantile_method"] = "approx_quantile (t-digest)"
    log.info(
        "%s: rows=%s pct_negative_age=%s median_views=%s",
        label,
        result["row_count"],
        result["pct_negative_observation_age"],
        result["median_views"],
    )
    return result


def observation_integrity_summary(audit: dict, threshold_pct: float = 1.0) -> dict:
    """Decide whether a partition can support a fixed-age label at all."""
    pct_negative = audit.get("pct_negative_observation_age") or 0.0
    pct_null = audit.get("pct_null_stats_ts") or 0.0
    reliable = pct_negative <= threshold_pct and pct_null <= threshold_pct
    return {
        "fixed_age_label_supported": bool(reliable),
        "pct_negative_observation_age": pct_negative,
        "pct_null_stats_timestamp": pct_null,
        "tolerance_pct": threshold_pct,
        "reason": (
            "Observation timestamps are internally consistent."
            if reliable
            else (
                f"{pct_negative:.2f}% of rows record engagement statistics at a timestamp "
                "earlier than publication. The observation clock is therefore not trustworthy "
                "and fixed-age (24h / 7d) targets cannot be constructed from this partition."
            )
        ),
    }
