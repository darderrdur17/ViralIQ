"""Tests for feature attribution.

The one-hot aggregation had its name/value zip order reversed, which silently
produced nonsense importance (and an exception for linear models). These tests pin
the contract so it cannot regress.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.explainability.attribution import (
    aggregate_to_inputs,
    global_importance,
    local_attributions,
    unwrap,
)


def test_aggregate_maps_values_to_the_right_names():
    """Names and values must not be swapped."""
    names = ["num__duration_s", "cat__country_US", "cat__country_ID"]
    values = np.array([0.5, 2.0, -1.0])
    totals = aggregate_to_inputs(values, names)
    assert totals["duration_s"] == pytest.approx(0.5)
    assert totals["country"] == pytest.approx(1.0)  # 2.0 + (-1.0), summed onto parent
    assert set(totals) == {"duration_s", "country"}


def test_aggregate_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        aggregate_to_inputs(np.array([1.0, 2.0]), ["a"])


def _tiny_training_frame(n: int = 120) -> tuple[pl.DataFrame, np.ndarray]:
    from datetime import datetime, timezone

    base = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "video_id": i,
                "posted_at": base.replace(hour=i % 24),
                "country": "US" if i % 3 else "ID",
                "language": "en",
                "type": "video",
                "duration_ms": int(rng.integers(5_000, 60_000)),
                "image_count": 0,
                "caption": "caption " * int(rng.integers(1, 4)),
                "hashtags": ["a", "b"][: int(rng.integers(0, 3))],
                "sound_id": int(rng.integers(0, 5)),
                "is_ad": 0,
                "branded_content": None,
                "is_shop_video": 0,
                "is_ai_generated": 0,
                "not_recommended_for_fyp": 0,
            }
        )
    df = pl.DataFrame(rows).with_columns(pl.col("posted_at").cast(pl.Datetime("us", "UTC")))
    y = (rng.random(n) < 0.15).astype(int)
    return df, y


def _fit_linear(df: pl.DataFrame, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    from src.feature_engineering.features import build_prepublication_features
    from src.models.encoder import FeatureEncoder

    featured = build_prepublication_features(df)
    enc = FeatureEncoder().fit(featured)
    X = enc.transform(featured)
    prep = ColumnTransformer(
        [
            ("num", "passthrough", list(enc.numeric)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(enc.categorical)),
        ]
    )
    pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X, y)
    return pipe, enc, featured


def test_global_importance_works_for_linear_pipeline():
    """Regression: linear models previously produced no importance at all."""
    df, y = _tiny_training_frame()
    pipe, _, _ = _fit_linear(df, y)
    ranked = global_importance(pipe)
    assert ranked, "linear pipeline must yield global importance"
    assert all("feature" in r and "importance" in r for r in ranked)
    assert sum(r["importance"] for r in ranked) == pytest.approx(1.0, abs=1e-6)


def test_local_attributions_aggregate_to_input_space():
    df, y = _tiny_training_frame()
    pipe, enc, featured = _fit_linear(df, y)
    X = enc.transform(featured)
    background = featured.head(50)
    result = local_attributions(pipe, X.head(1), background, enc)
    assert result is not None
    assert result["method"] == "shap_linear_aggregated"
    feats = {c["feature"] for c in result["top"]}
    # Must report input columns, never expanded dummy names.
    assert not any(f.startswith(("cat__", "num__")) for f in feats)
    assert "country" in feats or "duration_s" in feats


def test_unwrap_reaches_the_base_estimator():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    df, y = _tiny_training_frame()
    pipe, enc, featured = _fit_linear(df, y)
    X = enc.transform(featured)
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid").fit(X, y)
    assert unwrap(cal) is pipe
