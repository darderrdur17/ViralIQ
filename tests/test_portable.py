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
    spec = export_portable_spec(
        predictor.model,
        predictor.encoder,
        predictor.reference,
        predictor.thresholds,
        background,
        predictor.version,
        predictor.run_meta.get("started_at"),
    )
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
