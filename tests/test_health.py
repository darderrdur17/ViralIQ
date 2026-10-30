"""Tests for month selection and the health verdict.

`select_months` decides which partitions a refresh trains on. Getting it wrong is how
the degenerate three-month run happened, so it is worth pinning: it must reject a month
whose rows were observed past the age window, and it must never silently fall back to
"everything on disk".
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from src.monitoring.health import (
    MonthHealth,
    month_health,
    select_months,
)


def _mh(month: str, over: float, usable: bool | None = None) -> MonthHealth:
    return MonthHealth(
        month=month,
        file=f"x__{month}.parquet",
        rows=1000,
        median_age_hours=100.0,
        pct_over_window=over,
        pct_negative_age=0.0,
        usable=(over <= 25.0) if usable is None else usable,
        reason="test",
    )


def test_selects_newest_usable_months_in_order():
    months = [_mh("2026-08", 53.0), _mh("2026-09", 0.1), _mh("2026-10", 0.0)]
    selected, rejected = select_months(months)
    assert selected == ["2026-09", "2026-10"], "must exclude the over-window month"
    assert rejected == ["2026-08"]


def test_caps_at_max_months():
    months = [_mh(m, 0.0) for m in ("2026-06", "2026-07", "2026-08", "2026-09")]
    selected, _ = select_months(months, max_months=2)
    assert selected == ["2026-08", "2026-09"], "newest two, not oldest two"


def test_returns_empty_when_nothing_is_usable():
    months = [_mh("2026-08", 90.0), _mh("2026-07", 80.0)]
    selected, rejected = select_months(months)
    assert selected == []
    assert rejected == ["2026-07", "2026-08"]


def test_month_health_measures_observation_age_from_files():
    """A partition crawled long after publication must be flagged unusable."""
    with tempfile.TemporaryDirectory() as d:
        base = datetime(2026, 8, 1, tzinfo=timezone.utc)
        rows = 200

        def frame(age_hours: int) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "posted_at": [base + timedelta(minutes=i) for i in range(rows)],
                    "stats_updated_at": [
                        base + timedelta(minutes=i, hours=age_hours) for i in range(rows)
                    ],
                }
            ).with_columns(
                pl.col("posted_at").cast(pl.Datetime("us", "UTC")),
                pl.col("stats_updated_at").cast(pl.Datetime("us", "UTC")),
            )

        frame(900).write_parquet(Path(d) / "old__2026-08.parquet")   # past the window
        frame(5).write_parquet(Path(d) / "fresh__2026-09.parquet")   # inside it

        result = {m.month: m for m in month_health(Path(d))}
        assert result["2026-08"].usable is False
        assert result["2026-08"].pct_over_window == 100.0
        assert result["2026-09"].usable is True
        assert result["2026-09"].pct_over_window == 0.0


def test_month_health_tolerates_an_empty_directory():
    with tempfile.TemporaryDirectory() as d:
        assert month_health(Path(d)) == []
