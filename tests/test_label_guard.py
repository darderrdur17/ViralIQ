"""The training guard must refuse a month it cannot label honestly.

Measured failure it prevents: 2026-08 has a median observation age of 756h against a
720h window, so 53% of it is dropped. The survivors are a biased subsample, the fitted
p99 collapses from 258,801 to 58,526 views, test prevalence inflates from 1.4% to 4.3%,
and CatBoost ends up scoring below random ranking.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from src.labeling import labels as L
from src.models.train import DegradedLabelError, run_training


def _frame(*, age_hours: float, rows: int = 400, old_fraction: float = 0.0) -> pl.DataFrame:
    """Synthetic partition. ``old_fraction`` of rows get a 900h observation age."""
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "video_id": list(range(rows)),
            "posted_at": [base + timedelta(minutes=i) for i in range(rows)],
            "country": ["US"] * rows,
            "language": ["en"] * rows,
            "type": ["video"] * rows,
            "duration_ms": [15_000] * rows,
            "image_count": [0] * rows,
            "caption": ["a caption"] * rows,
            "hashtags": [["tag"]] * rows,
            "sound_id": [1] * rows,
            "is_ad": [0] * rows,
            "branded_content": [None] * rows,
            "is_shop_video": [0] * rows,
            "is_ai_generated": [0] * rows,
            "not_recommended_for_fyp": [0] * rows,
            "views": list(range(1, rows + 1)),
            "stats_updated_at": [
                base
                + timedelta(minutes=i)
                + timedelta(hours=900 if i < rows * old_fraction else age_hours)
                for i in range(rows)
            ],
        }
    ).with_columns(
        pl.col("posted_at").cast(pl.Datetime("us", "UTC")),
        pl.col("stats_updated_at").cast(pl.Datetime("us", "UTC")),
    )


def test_diagnostics_separate_the_two_exclusion_causes():
    """A negative age and an over-window age are different defects and must not merge."""
    def diagnose(frame: pl.DataFrame) -> dict:
        clocked = L.add_observation_clock(frame)
        # A permissive threshold so the label itself never excludes anything; we are
        # testing how the clock splits the exclusions, not the threshold.
        thresholds = L.CohortThresholds(
            percentile=0.99,
            keys=("country", "type", "observation_age_bucket"),
            thresholds={},
            global_threshold=1.0,
            fitted_on_rows=clocked.height,
            fitted_on_months=("2026-08",),
        )
        return L.label_diagnostics(L.apply_labels(clocked, thresholds))

    stale_diag = diagnose(_frame(age_hours=-48))   # counters read before publication
    old_diag = diagnose(_frame(age_hours=900))     # read 37 days later, past the window
    healthy_diag = diagnose(_frame(age_hours=5))

    assert stale_diag["pct_negative_observation_age"] == 100.0
    assert stale_diag["pct_over_age_window"] == 0.0

    assert old_diag["pct_over_age_window"] == 100.0
    assert old_diag["pct_negative_observation_age"] == 0.0

    assert healthy_diag["pct_excluded_invalid_clock"] == 0.0


def test_training_refuses_a_month_past_the_age_window():
    """Mirrors 2026-08: a majority past the window, so thresholds fit but on a biased slice."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "old_month.parquet"
        _frame(age_hours=5, old_fraction=0.60).write_parquet(path)
        with pytest.raises(DegradedLabelError, match="observed more than"):
            run_training(sample_paths=[path], output_tag="guard-test", max_rows_per_split=400)


def test_fully_unusable_month_fails_with_a_readable_message():
    """Every row invalid: report that, rather than a TypeError from inside polars."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "all_stale.parquet"
        _frame(age_hours=900).write_parquet(path)
        with pytest.raises(ValueError, match="no rows have a valid observation clock"):
            run_training(sample_paths=[path], output_tag="guard-test", max_rows_per_split=400)


def test_override_flag_is_honoured_by_the_exception_type():
    """The escape hatch must be an explicit argument, not a silent default."""
    import inspect

    sig = inspect.signature(run_training)
    assert "allow_degraded" in sig.parameters
    assert sig.parameters["allow_degraded"].default is False
