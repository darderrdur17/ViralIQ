"""Tests for label construction and the leakage guard."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.labeling import labels as L

T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def _frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("posted_at").cast(pl.Datetime("us", "UTC")),
        pl.col("stats_updated_at").cast(pl.Datetime("us", "UTC")),
    )


def test_negative_observation_age_is_flagged_invalid():
    df = _frame(
        [
            {"video_id": 1, "country": "US", "type": "video", "posted_at": T0,
             "stats_updated_at": T0 + timedelta(hours=5), "views": 100},
            # statistics recorded before publication: impossible
            {"video_id": 2, "country": "US", "type": "video", "posted_at": T0,
             "stats_updated_at": T0 - timedelta(days=7), "views": 100},
        ]
    )
    out = L.add_observation_clock(df)
    assert out["observation_valid"].to_list() == [True, False]
    assert out["observation_age_hours"][1] < 0


def test_labels_never_fabricate_fixed_age_targets():
    df = _frame(
        [
            {"video_id": i, "country": "US", "type": "video", "posted_at": T0,
             "stats_updated_at": T0 + timedelta(hours=3), "views": v}
            for i, v in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90, 1000])
        ]
    )
    df = L.add_observation_clock(df)
    thr = L.fit_cohort_thresholds(df, months=("2026-09",))
    out = L.apply_labels(df, thr)

    assert out["viral_24h"].null_count() == out.height
    assert out["viral_7d"].null_count() == out.height
    assert out["future_view_count"].null_count() == out.height


def test_threshold_is_cohort_relative_not_global():
    # Cohorts must exceed min_cohort_size (500) for a per-cohort threshold to be fitted.
    df = _frame(
        [
            {"video_id": i, "country": c, "type": "video", "posted_at": T0,
             "stats_updated_at": T0 + timedelta(hours=2), "views": v}
            for i, (c, v) in enumerate(
                [("US", 100_000)] * 600 + [("ID", 900)] * 600
            )
        ]
    )
    df = L.add_observation_clock(df)
    thr = L.fit_cohort_thresholds(df, months=("2026-09",))
    out = L.apply_labels(df, thr)

    us = out.filter(pl.col("country") == "US")
    id_ = out.filter(pl.col("country") == "ID")
    # Each cohort gets its own p99 threshold, so the low-reach cohort is not
    # labelled negative purely because of absolute view magnitude.
    assert len(thr.thresholds) == 2
    assert us["viral_view_threshold"].unique().to_list()[0] == pytest.approx(100_000)
    assert id_["viral_view_threshold"].unique().to_list()[0] == pytest.approx(900)


def test_small_cohorts_fall_back_to_global_threshold():
    df = _frame(
        [
            {"video_id": i, "country": "US", "type": "video", "posted_at": T0,
             "stats_updated_at": T0 + timedelta(hours=2), "views": v}
            for i, v in enumerate(range(1, 101))
        ]
    )
    df = L.add_observation_clock(df)
    thr = L.fit_cohort_thresholds(df, months=("2026-09",))
    # 100 rows is below min_cohort_size, so no per-cohort threshold is stored.
    assert thr.thresholds == {}
    out = L.apply_labels(df, thr)
    assert out["viral_view_threshold"].unique().to_list()[0] == pytest.approx(thr.global_threshold)


def test_leakage_guard_rejects_performance_columns():
    with pytest.raises(ValueError, match="Leakage guard"):
        L.assert_no_performance_features(["caption_len", "views"])
    L.assert_no_performance_features(["caption_len", "duration_s"])


def test_invalid_clock_rows_are_excluded_from_positives():
    df = _frame(
        [
            {"video_id": i, "country": "US", "type": "video", "posted_at": T0,
             "stats_updated_at": T0 + timedelta(hours=2), "views": 10}
            for i in range(5)
        ]
        + [
            {"video_id": 99, "country": "US", "type": "video", "posted_at": T0,
             "stats_updated_at": T0 - timedelta(days=3), "views": 10_000_000}
        ]
    )
    df = L.add_observation_clock(df)
    thr = L.fit_cohort_thresholds(df, months=("2026-09",))
    out = L.apply_labels(df, thr)
    invalid = out.filter(~pl.col("observation_valid"))
    assert invalid["viral_at_snapshot"].sum() == 0
    assert invalid["label_basis"][0] == "excluded_invalid_observation_clock"
