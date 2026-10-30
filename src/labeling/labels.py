"""Target definition and label engineering.

Two rules govern this module:

1. A label may only use information from *after* the prediction horizon. It is
   never a feature.
2. If the dataset cannot support a fixed-age label, we say so instead of
   inventing one. Fixed-age labels (``viral_24h`` / ``viral_7d``) are emitted as
   NULL with an explicit reason unless a valid observation clock exists.

The virality threshold is cohort-relative (percentile within a peer group), never
a hard-coded global view count.
"""
from __future__ import annotations

import dataclasses
from typing import Iterable

import polars as pl

from src.common.config import SETTINGS
from src.common.logging import get_logger

log = get_logger("labels")

OBSERVATION_AGE = (pl.col("stats_updated_at").cast(pl.Datetime("us", "UTC")) -
                   pl.col("posted_at").cast(pl.Datetime("us", "UTC"))).dt.total_hours()

# Columns that encode realised performance and must never become features.
PERFORMANCE_COLUMNS: frozenset[str] = frozenset(
    {"views", "likes", "comments", "shares", "saves", "downloads", "engagement_rate"}
)


@dataclasses.dataclass(frozen=True)
class CohortThresholds:
    """Frozen, train-window-derived virality thresholds."""

    percentile: float
    keys: tuple[str, ...]
    thresholds: dict[str, float]
    global_threshold: float
    fitted_on_rows: int
    fitted_on_months: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "percentile": self.percentile,
            "keys": list(self.keys),
            "global_threshold": self.global_threshold,
            "cohort_count": len(self.thresholds),
            "fitted_on_rows": self.fitted_on_rows,
            "fitted_on_months": list(self.fitted_on_months),
            "thresholds": self.thresholds,
        }


def _cohort_key_expr(keys: Iterable[str]) -> pl.Expr:
    parts = [pl.col(k).cast(pl.Utf8).fill_null("__na__") for k in keys]
    return pl.concat_str(parts, separator="|")


def add_observation_clock(df: pl.DataFrame, cfg=None) -> pl.DataFrame:
    """Attach observation age, an age bucket, and a validity flag.

    ``observation_valid`` is False when the recorded statistics predate
    publication (impossible) or exceed the configured horizon.

    ``observation_age_bucket`` matters for fairness. A video measured 3 hours after
    publication and one measured 20 days after publication are not comparable: the
    second has had far longer to accumulate views. Rather than discard most of the
    corpus or compare them naively, the bucket becomes part of the peer group, so a
    video is only ever compared against videos observed at a similar age.
    """
    cfg = cfg or SETTINGS.labels
    out = df.with_columns(OBSERVATION_AGE.alias("observation_age_hours"))
    out = out.with_columns(
        (
            (pl.col("observation_age_hours") >= cfg.min_observation_age_hours)
            & (pl.col("observation_age_hours") <= cfg.max_observation_age_hours)
            & pl.col("stats_updated_at").is_not_null()
        ).alias("observation_valid")
    )
    out = out.with_columns(
        pl.when(pl.col("observation_age_hours") < 0).then(pl.lit("invalid"))
        .when(pl.col("observation_age_hours") <= 6).then(pl.lit("0-6h"))
        .when(pl.col("observation_age_hours") <= 24).then(pl.lit("6-24h"))
        .when(pl.col("observation_age_hours") <= 72).then(pl.lit("1-3d"))
        .when(pl.col("observation_age_hours") <= 168).then(pl.lit("3-7d"))
        .when(pl.col("observation_age_hours") <= 720).then(pl.lit("7-30d"))
        .otherwise(pl.lit("over30d"))
        .alias("observation_age_bucket")
    )
    return out


def fit_cohort_thresholds(
    train_df: pl.DataFrame,
    months: tuple[str, ...],
    cfg=None,
) -> CohortThresholds:
    """Fit virality thresholds on the *training window only*.

    Thresholds are frozen here and applied unchanged to validation and test
    splits, so the test-set label definition cannot drift toward the test
    distribution.
    """
    cfg = cfg or SETTINGS.labels
    eligible = train_df.filter(pl.col("observation_valid"))
    if eligible.height == 0:
        # Without this the quantile below returns None and fails as a confusing
        # TypeError deep inside polars, which says nothing about the real cause.
        raise ValueError(
            "no rows have a valid observation clock; thresholds cannot be fitted. "
            "Check the observation-age window against this partition's crawl timestamps."
        )
    key = _cohort_key_expr(cfg.cohort_keys)
    eligible = eligible.with_columns(key.alias("_cohort"))

    global_threshold = float(eligible["views"].quantile(cfg.percentile))

    grouped = (
        eligible.group_by("_cohort")
        .agg(
            pl.len().alias("n"),
            pl.col("views").quantile(cfg.percentile).alias("thr"),
        )
        .filter(pl.col("n") >= cfg.min_cohort_size)
    )
    thresholds = {
        row["_cohort"]: float(row["thr"]) for row in grouped.iter_rows(named=True)
    }
    log.info(
        "fitted %d cohort thresholds at p%.1f (global=%.0f views) on %d rows",
        len(thresholds),
        cfg.percentile * 100,
        global_threshold,
        eligible.height,
    )
    return CohortThresholds(
        percentile=cfg.percentile,
        keys=tuple(cfg.cohort_keys),
        thresholds=thresholds,
        global_threshold=global_threshold,
        fitted_on_rows=eligible.height,
        fitted_on_months=months,
    )


def apply_labels(df: pl.DataFrame, thresholds: CohortThresholds) -> pl.DataFrame:
    """Apply frozen thresholds and derive label columns.

    Emits:
      * ``viral_at_snapshot`` — the usable retrospective label
      * ``performance_percentile`` — within-cohort percentile of observed views
      * ``viral_24h`` / ``viral_7d`` — NULL, because this dataset has no repeated
        observations and therefore cannot support fixed-age outcomes
      * ``label_basis`` — human-readable provenance for every row
    """
    key = _cohort_key_expr(thresholds.keys)
    out = df.with_columns(key.alias("_cohort"))

    thr_expr = pl.col("_cohort").replace_strict(
        thresholds.thresholds, default=thresholds.global_threshold, return_dtype=pl.Float64
    )
    out = out.with_columns(thr_expr.alias("viral_view_threshold"))

    out = out.with_columns(
        (
            pl.col("observation_valid")
            & (pl.col("views") >= pl.col("viral_view_threshold"))
        ).fill_null(False).alias("viral_at_snapshot")
    )

    # Within-cohort percentile of the observed snapshot count.
    out = out.with_columns(
        (
            pl.col("views").rank("average").over("_cohort")
            / pl.len().over("_cohort")
            * 100.0
        ).alias("performance_percentile")
    )

    out = out.with_columns(
        pl.lit(None, dtype=pl.Boolean).alias("viral_24h"),
        pl.lit(None, dtype=pl.Boolean).alias("viral_7d"),
        pl.lit(None, dtype=pl.Int64).alias("future_view_count"),
        pl.when(pl.col("observation_valid"))
        .then(pl.lit("snapshot_top1pct_cohort_relative"))
        .otherwise(pl.lit("excluded_invalid_observation_clock"))
        .alias("label_basis"),
    )
    return out.drop("_cohort")


def label_diagnostics(df: pl.DataFrame) -> dict:
    """Summarise label prevalence and the exclusion it required."""
    n = df.height
    if n == 0:
        return {"rows": 0}
    valid = df.filter(pl.col("observation_valid"))
    positives = int(valid["viral_at_snapshot"].sum() or 0)

    # Two different defects hide inside "invalid clock", and they matter differently:
    # a negative age is a stale crawl batch spread across the month, whereas rows past
    # the age window are a *biased subsample* of the month's reach distribution, which
    # drags the fitted threshold with them. Report them separately.
    cfg = SETTINGS.labels
    pct_negative = round(
        100.0 * float((df["observation_age_hours"] < 0).sum()) / max(n, 1), 3
    )
    pct_over_window = round(
        100.0
        * float((df["observation_age_hours"] > cfg.max_observation_age_hours).sum())
        / max(n, 1),
        3,
    )
    return {
        "rows": n,
        "rows_observation_valid": valid.height,
        "rows_excluded_invalid_clock": n - valid.height,
        "pct_excluded_invalid_clock": round(100 * (n - valid.height) / n, 3),
        "pct_negative_observation_age": pct_negative,
        "pct_over_age_window": pct_over_window,
        "positives": positives,
        "prevalence_valid": round(positives / valid.height, 6) if valid.height else None,
        "prevalence_all_rows": round(positives / n, 6),
        "fixed_age_labels_available": False,
        "fixed_age_label_reason": (
            "The source dataset stores a single engagement snapshot per video with no "
            "repeated observations. viral_24h and viral_7d are therefore NULL rather "
            "than estimated."
        ),
    }


def assert_no_performance_features(feature_names: Iterable[str]) -> None:
    """Hard guard: realised engagement must never enter a pre-publication model."""
    leaked = PERFORMANCE_COLUMNS.intersection(set(feature_names))
    if leaked:
        raise ValueError(
            f"Leakage guard tripped: performance columns used as features: {sorted(leaked)}"
        )
