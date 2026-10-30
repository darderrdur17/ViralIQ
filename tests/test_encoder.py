"""Tests for the feature encoder's handling of unseen categories."""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from src.feature_engineering.features import build_prepublication_features
from src.models.encoder import UNSEEN, FeatureEncoder

NOW = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)


def _raw(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("posted_at").cast(pl.Datetime("us", "UTC")))


def _row(**kw) -> dict:
    base = {
        "video_id": 1,
        "posted_at": NOW,
        "country": "US",
        "language": "en",
        "type": "video",
        "duration_ms": 15_000,
        "image_count": 0,
        "caption": "test caption",
        "hashtags": ["a", "b"],
        "sound_id": 5,
        "is_ad": 0,
        "branded_content": None,
        "is_shop_video": 0,
        "is_ai_generated": 0,
        "not_recommended_for_fyp": 0,
    }
    base.update(kw)
    return base


def test_unseen_category_maps_to_sentinel():
    train = _raw([_row(video_id=i, country="US") for i in range(50)])
    enc = FeatureEncoder().fit(build_prepublication_features(train))

    unseen = _raw([_row(country="ZZ")])
    X = enc.transform(build_prepublication_features(unseen))
    assert str(X["country"].iloc[0]) == UNSEEN


def test_encoder_produces_stable_feature_count():
    train = _raw([_row(video_id=i) for i in range(20)])
    enc = FeatureEncoder().fit(build_prepublication_features(train))
    X = enc.transform(build_prepublication_features(train))
    assert list(X.columns) == enc.feature_names
    assert X.shape[1] == len(enc.feature_names)


def test_missing_numeric_filled_with_median():
    train = _raw([_row(video_id=i, duration_ms=d) for i, d in enumerate([1000, 2000, 3000])])
    featured = build_prepublication_features(train)
    enc = FeatureEncoder().fit(featured)

    bad = _raw([_row(duration_ms=None)])
    X = enc.transform(build_prepublication_features(bad))
    assert X["duration_s"].iloc[0] == enc.fill_values["duration_s"]


def test_encoder_roundtrip():
    import tempfile
    from pathlib import Path

    train = _raw([_row(video_id=i) for i in range(10)])
    enc = FeatureEncoder().fit(build_prepublication_features(train))
    path = Path(tempfile.mkdtemp()) / "encoder.json"
    enc.save(path)
    restored = FeatureEncoder.load(path)
    assert restored.vocab == enc.vocab
    assert restored.fill_values == enc.fill_values
