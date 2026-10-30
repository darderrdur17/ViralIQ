"""Tests for the external-file importer.

These cover the failure that mattered most: the Parquet branch passed a CSV-only
argument to the reader, so every Parquet import failed. They also pin the
normalisation and no-overwrite guarantees.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.ingestion.importer import (
    ImportRejected,
    load_and_normalise,
    store_sample,
)


def _base_rows(n: int = 3) -> dict:
    base = datetime(2026, 10, 1, tzinfo=timezone.utc)
    return {
        "video_id": list(range(n)),
        "posted_at": [base.replace(hour=i) for i in range(n)],
        "country": ["US"] * n,
        "language": ["en"] * n,
        "type": ["video"] * n,
        "duration_ms": [10_000] * n,
        "caption": ["a caption"] * n,
        "sound_id": [1] * n,
        "views": [100] * n,
        "likes": [1] * n,
        "comments": [0] * n,
        "shares": [0] * n,
        "saves": [0] * n,
        "stats_updated_at": [base.replace(hour=i + 5) for i in range(n)],
    }


def _parquet(rows: dict, tmp: Path, name: str = "in.parquet") -> Path:
    p = tmp / name
    pl.DataFrame(rows).write_parquet(p)
    return p


def test_parquet_import_reads_successfully():
    """Regression: the Parquet branch previously raised TypeError for every file."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = _base_rows()
        rows["hashtags"] = [["foodtok"], ["a", "b"], []]
        df, result = load_and_normalise(_parquet(rows, tmp))
        assert df.height == 3
        assert result.original_format == "parquet"


def test_csv_hashtags_are_split_into_lists():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = _base_rows()
        rows["hashtags"] = ["foodtok,recipe", "solo", ""]
        csv = tmp / "in.csv"
        pl.DataFrame(rows).write_csv(csv)

        df, result = load_and_normalise(csv)
        assert df.schema["hashtags"] == pl.List(pl.String)
        assert df["hashtags"][0].to_list() == ["foodtok", "recipe"]
        assert df["hashtags"][2].to_list() == []
        assert any("hashtags" in n for n in result.normalisations)


def test_csv_timestamps_and_numbers_are_coerced():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = _base_rows()
        rows["hashtags"] = ["x", "y", "z"]
        rows["posted_at"] = ["2026-10-01T00:00:00Z", "2026-10-01T01:00:00Z", "2026-10-01T02:00:00Z"]
        rows["duration_ms"] = ["10000", "20000", "30000"]
        csv = tmp / "in.csv"
        pl.DataFrame(rows).write_csv(csv)

        df, _ = load_and_normalise(csv)
        assert isinstance(df.schema["posted_at"], pl.Datetime)
        assert df.schema["duration_ms"] in (pl.Int64, pl.Int32)


def test_missing_required_columns_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        p = tmp / "bad.parquet"
        pl.DataFrame({"foo": [1]}).write_parquet(p)
        with pytest.raises(ImportRejected, match="missing required columns"):
            load_and_normalise(p)


def test_unsupported_extension_message_is_readable():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        p = tmp / "noext"
        p.write_text("hello")
        with pytest.raises(ImportRejected, match=r"Unsupported file type: \(no extension\)"):
            load_and_normalise(p)


def test_empty_file_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        p = tmp / "empty.parquet"
        pl.DataFrame(schema={"video_id": pl.Int64}).write_parquet(p)
        with pytest.raises(ImportRejected):
            load_and_normalise(p)


def test_store_sample_refuses_to_overwrite():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        target = tmp / "samples"
        rows = _base_rows()
        rows["hashtags"] = [["a"], ["b"], ["c"]]
        df, _ = load_and_normalise(_parquet(rows, tmp))
        first = store_sample(df, Path("/fake/input.parquet"), target)
        assert first.exists()
        with pytest.raises(ImportRejected, match="refusing to overwrite"):
            store_sample(df, Path("/fake/input.parquet"), target)


def test_stored_sample_is_parquet_so_analytics_can_see_it():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = _base_rows()
        rows["hashtags"] = ["a,b", "c", "d"]
        csv = tmp / "export.csv"
        pl.DataFrame(rows).write_csv(csv)
        df, _ = load_and_normalise(csv)
        out = store_sample(df, csv, tmp / "samples")
        assert out.suffix == ".parquet", "imports must land as Parquet or analytics will skip them"
        assert pl.read_parquet(out).height == 3
