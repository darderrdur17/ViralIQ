"""Parity between the portable surrogate and the real fitted pipeline.

The serverless deployment cannot ship scikit-learn, so it runs a portable affine
surrogate exported from the fitted pipeline. That is only acceptable if it produces
the same numbers — these tests are the guarantee.
"""
from __future__ import annotations

import json

import polars as pl
import pytest

from src.feature_engineering.features import build_prepublication_features
from src.inference.portable import PortableModel, export_portable_spec

CASES = [
    {"caption": "three ingredient pasta hack nobody talks about",
     "hashtags": ["foodtok", "recipe"], "duration_ms": 21000, "country": "US", "language": "en"},
    {"caption": "gym motivation no excuses", "hashtags": ["fitness"],
     "duration_ms": 9000, "country": "ID", "language": "id"},
    {"caption": "", "hashtags": [], "duration_ms": 0, "country": "ZZ", "language": "xx"},
    {"caption": "unboxing asmr 4k", "hashtags": [], "duration_ms": 120000,
     "country": "BR", "language": "pt", "sound_id": 123456789},
    {"caption": "a" * 300, "hashtags": ["a", "b", "c", "d"], "duration_ms": 60000,
     "country": "JP", "language": "ja", "is_ad": True},
    {"caption": "short", "hashtags": ["x"], "duration_ms": 3000, "country": "US",
     "language": "en", "image_count": 4},
]


@pytest.fixture(scope="module")
def harness():
    try:
        from src.inference.predictor import Predictor

        predictor = Predictor()
    except Exception as exc:  # pragma: no cover - depends on local artefacts
        pytest.skip(f"no trained model available: {exc}")

    background = pl.read_parquet(predictor.run_dir / "reference_features.parquet").select(
        predictor.encoder.feature_names
    )
    try:
        spec = export_portable_spec(
            predictor.model,
            predictor.encoder,
            predictor.reference,
            predictor.thresholds,
            background,
            predictor.version,
            predictor.run_meta.get("started_at"),
        )
    except TypeError as exc:
        pytest.skip(f"active champion is not linear, so the surrogate does not apply: {exc}")
    portable = PortableModel(spec, predictor.sound_stats)
    return predictor, portable, spec


def _features(predictor, payload):
    row = predictor._row_from_payload(payload)
    featured = build_prepublication_features(row, predictor.sound_stats)
    return featured, {name: featured[name][0] for name in predictor.encoder.feature_names}


def test_portable_probabilities_match_sklearn(harness):
    predictor, portable, _ = harness
    for payload in CASES:
        featured, raw = _features(predictor, payload)
        X = predictor.encoder.transform(featured)
        expected = float(predictor.model.predict_proba(X)[:, 1][0])
        actual = portable.probability(raw)
        assert abs(expected - actual) < 1e-7, (
            f"parity drift for {payload['country']}/{payload['language']}: "
            f"sklearn={expected:.12f} portable={actual:.12f}"
        )


def test_portable_handles_unseen_categories(harness):
    """An unknown country must route through the same '__unseen__' bucket."""
    predictor, portable, _ = harness
    featured, raw = _features(predictor, CASES[2])
    X = predictor.encoder.transform(featured)
    assert abs(float(predictor.model.predict_proba(X)[:, 1][0]) - portable.probability(raw)) < 1e-7


def test_portable_spec_is_serialisable_and_small(harness):
    _, _, spec = harness
    blob = json.dumps(spec)
    assert len(blob) < 200_000, "portable spec must stay small enough for a function bundle"
    restored = PortableModel(json.loads(blob), None)
    assert restored.feature_names == spec["feature_names"]


def test_attributions_rank_the_same_top_feature(harness):
    """Portable SHAP and sklearn SHAP should agree on the dominant driver."""
    predictor, portable, _ = harness
    payload = CASES[0]
    featured, raw = _features(predictor, payload)
    X = predictor.encoder.transform(featured)

    from src.explainability.attribution import local_attributions

    reference = pl.read_parquet(predictor.run_dir / "reference_features.parquet").select(
        predictor.encoder.feature_names
    )
    sklearn_attr = local_attributions(predictor.model, X, reference, predictor.encoder)
    portable_attr = portable.attributions(raw)

    assert sklearn_attr and sklearn_attr["top"], "sklearn attribution produced nothing"
    assert portable_attr, "portable attribution produced nothing"
    assert portable_attr[0]["feature"] == sklearn_attr["top"][0]["feature"]


def test_portable_reference_and_thresholds_are_present(harness):
    _, portable, _ = harness
    assert portable.reference.get("view_quantiles"), "view quantiles needed for percentile"
    assert portable.reference.get("bands"), "score bands needed for expected views"
    assert portable.thresholds.get("global_threshold"), "threshold needed for the UI"
    assert portable.thresholds.get("keys"), "cohort keys needed for the label description"


def test_export_roundtrip_on_a_linear_pipeline():
    """Covers the surrogate mechanism regardless of which model is active.

    The tests above need a linear champion; when the active model is a forest they
    skip. This one always runs, so CI cannot lose coverage of the export logic just
    because a retrain changed the champion.
    """
    from datetime import datetime, timezone

    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from src.feature_engineering.features import build_prepublication_features
    from src.models.encoder import FeatureEncoder

    rng = np.random.default_rng(7)
    base = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(200):
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
    frame = pl.DataFrame(rows).with_columns(
        pl.col("posted_at").cast(pl.Datetime("us", "UTC"))
    )
    featured = build_prepublication_features(frame)
    encoder = FeatureEncoder().fit(featured)
    X = encoder.transform(featured)
    y = (rng.random(len(rows)) < 0.15).astype(int)

    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), list(encoder.numeric)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(encoder.categorical)),
        ]
    )
    pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=500))])
    # FrozenEstimator does not fit its wrapped estimator, so fit it first — the same
    # order the training pipeline uses.
    pipe.fit(X, y)
    model = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid").fit(X, y)

    spec = export_portable_spec(
        model, encoder, {}, {"global_threshold": 1000.0, "percentile": 0.99, "keys": ["country"]},
        featured.select(encoder.feature_names), "test@roundtrip", None,
    )
    portable = PortableModel(spec, None)

    for i in range(5):
        raw = {name: featured[name][i] for name in encoder.feature_names}
        expected = float(model.predict_proba(X.iloc[[i]])[:, 1][0])
        assert abs(portable.probability(raw) - expected) < 1e-7
