"""Tests for the chronological split.

The split is the single most important correctness guarantee in the project: if it
leaks, every reported metric is invalid. These tests pin the calendar behaviour,
including the 3+ month path that a two-month run never exercises.
"""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from src.models.train import temporal_slices


def _frame(months: list[str], rows_per_month: int = 20) -> pl.DataFrame:
    rows = []
    for m in months:
        year, month = (int(x) for x in m.split("-"))
        for i in range(rows_per_month):
            day = 1 + (i % 27)
            rows.append({"posted_at": datetime(year, month, day, 12, tzinfo=timezone.utc)})
    return pl.DataFrame(rows).with_columns(pl.col("posted_at").cast(pl.Datetime("us", "UTC")))


def _month(series: pl.Series) -> set[str]:
    return set(series.dt.strftime("%Y-%m").unique().to_list())


def test_two_months_train_then_halves_of_next():
    df = _frame(["2026-09", "2026-10"])
    s = temporal_slices(df)
    assert _month(s["train"]["posted_at"]) == {"2026-09"}
    assert _month(s["val"]["posted_at"]) == {"2026-10"}
    assert _month(s["test"]["posted_at"]) == {"2026-10"}
    assert s["recent"].height == 0
    # calibration and test must not share a row
    assert s["val"]["posted_at"].max() < s["test"]["posted_at"].min()


def test_three_months_uses_third_month_as_forward_holdout():
    df = _frame(["2026-08", "2026-09", "2026-10"])
    s = temporal_slices(df)
    assert _month(s["train"]["posted_at"]) == {"2026-08"}
    assert _month(s["test"]["posted_at"]) == {"2026-09"}
    assert _month(s["recent"]["posted_at"]) == {"2026-10"}
    assert s["recent"].height > 0, "third month must be used as the forward holdout"


def test_four_months_holdout_is_the_month_after_test():
    """Regression: months[2] was previously skipped in favour of months[3]."""
    df = _frame(["2026-07", "2026-08", "2026-09", "2026-10"])
    s = temporal_slices(df)
    assert _month(s["train"]["posted_at"]) == {"2026-07"}
    assert _month(s["test"]["posted_at"]) == {"2026-08"}
    assert _month(s["recent"]["posted_at"]) == {"2026-09"}, (
        "forward holdout must be the month immediately after the test month"
    )


def test_single_month_splits_chronologically_without_overlap():
    df = _frame(["2026-10"], rows_per_month=60)
    s = temporal_slices(df)
    assert s["train"].height > 0 and s["val"].height > 0 and s["test"].height > 0
    assert s["train"]["posted_at"].max() <= s["val"]["posted_at"].min()
    assert s["val"]["posted_at"].max() < s["test"]["posted_at"].min()


def test_all_splits_are_non_empty_and_disjoint():
    df = _frame(["2026-08", "2026-09", "2026-10"])
    s = temporal_slices(df)
    total = sum(part.height for part in s.values())
    assert total == df.height, "every row must land in exactly one split"

    keys = {
        name: set(part["posted_at"].to_list()) for name, part in s.items() if part.height
    }
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = keys[a] & keys[b]
            assert not overlap, f"{a} and {b} share timestamps"
