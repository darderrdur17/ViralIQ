"""Portable inference for serverless deployment.

Vercel's serverless bundle limit (250 MB unzipped) cannot hold scikit-learn plus
SciPy, pandas and the rest of the training stack. Rather than deploy a broken
subset, this module exports the *behaviour* of the fitted model into a small,
dependency-light artefact.

How it works
------------
The champion is a linear pipeline: standardise numerics, one-hot categoricals,
linear combination, Platt scaling. Every stage is affine or an indicator, so the
whole pipeline is an affine function of the raw feature vector:

    decision(x) = d0 + Σ cᵢ·xᵢ + Σ c_col,value·[x_col == value]
    probability = sigmoid(-(A·decision + B))

The coefficients are recovered by *probing the fitted pipeline* — feeding one row per
feature and taking the difference in decision value — instead of reading sklearn's
private fitted attributes, which have moved between versions. Probing is exact for an
affine model and does not depend on library internals.

`tests/test_portable.py` asserts numerical parity with the real predictor, so this
cannot silently drift.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common.logging import get_logger

log = get_logger("portable")

BASELINE_CATEGORY = "__unseen__"


def _sigmoid(z: np.ndarray | float):
    return 1.0 / (1.0 + np.exp(-z))


class PortableModel:
    """Affine surrogate of the fitted pipeline, evaluated with NumPy only."""

    def __init__(self, spec: dict, sound_stats=None):
        self.feature_names: list[str] = spec["feature_names"]
        self.categorical: list[str] = spec["categorical"]
        self.numeric: list[str] = spec["numeric"]
        self.fill_values: dict[str, float] = spec["fill_values"]
        self.vocab: dict[str, list[str]] = spec["vocab"]
        self.intercept: float = spec["intercept"]
        self.coefficients: dict[str, float] = spec["coefficients"]
        self.categorical_coefficients: dict[str, dict[str, float]] = spec[
            "categorical_coefficients"
        ]
        self.calibration: dict[str, float] = spec["calibration"]
        self.background_means: dict[str, float] = spec["background_means"]
        self.reference: dict = spec["reference"]
        self.thresholds: dict = spec["thresholds"]
        self.model_version: str = spec["model_version"]
        self.trained_at: str | None = spec.get("trained_at")
        self.sound_stats = sound_stats

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: Path, sound_stats_path: Path | None = None):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        stats = None
        if sound_stats_path and Path(sound_stats_path).exists():
            import polars as pl

            stats = pl.read_parquet(sound_stats_path)
        return cls(spec, stats)

    # ------------------------------------------------------------------ #
    def _vector(self, features: dict[str, Any]) -> np.ndarray:
        """Raw feature vector with the same imputation the encoder applies."""
        vec = np.empty(len(self.feature_names), dtype=float)
        for i, name in enumerate(self.feature_names):
            if name in self.numeric:
                value = features.get(name)
                vec[i] = self.fill_values.get(name, 0.0) if value is None else float(value)
            else:
                vec[i] = 0.0  # categoricals handled separately
        return vec

    def decision(self, features: dict[str, Any]) -> float:
        vec = self._vector(features)
        total = self.intercept
        for i, name in enumerate(self.feature_names):
            if name in self.numeric:
                total += self.coefficients.get(name, 0.0) * vec[i]
        for col in self.categorical:
            value = features.get(col)
            key = value if value in self.vocab.get(col, []) else BASELINE_CATEGORY
            total += self.categorical_coefficients.get(col, {}).get(key, 0.0)
        return float(total)

    def probability(self, features: dict[str, Any]) -> float:
        a = self.calibration["a"]
        b = self.calibration["b"]
        return float(_sigmoid(-(a * self.decision(features) + b)))

    # ------------------------------------------------------------------ #
    def attributions(self, features: dict[str, Any], top_n: int = 8) -> list[dict]:
        """Linear SHAP: φᵢ = cᵢ · (xᵢ − E[xᵢ]) against the background distribution."""
        contributions: dict[str, float] = {}
        for name in self.numeric:
            value = features.get(name)
            value = self.fill_values.get(name, 0.0) if value is None else float(value)
            mean = self.background_means.get(name, 0.0)
            contributions[name] = self.coefficients.get(name, 0.0) * (value - mean)

        for col in self.categorical:
            value = features.get(col)
            key = value if value in self.vocab.get(col, []) else BASELINE_CATEGORY
            coefficients = self.categorical_coefficients.get(col, {})
            chosen = coefficients.get(key, 0.0)
            expected = sum(
                coefficients.get(v, 0.0) * self.background_means.get(f"{col}::{v}", 0.0)
                for v in self.vocab.get(col, [])
            )
            contributions[col] = chosen - expected

        ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:top_n]
        return [
            {
                "feature": name,
                "value": features.get(name),
                "contribution": round(value, 5),
                "direction": "increases" if value > 0 else "decreases",
            }
            for name, value in ranked
        ]


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def _probe_frame(rows: list[dict], encoder) -> Any:
    """Build a pandas frame the fitted ColumnTransformer will accept."""
    import polars as pl

    frame = pl.DataFrame(rows).select(encoder.feature_names)
    return encoder.transform(frame)


def export_portable_spec(model, encoder, reference: dict, thresholds: dict,
                         background_features, model_version: str,
                         trained_at: str | None) -> dict:
    """Derive the affine surrogate by probing the fitted pipeline.

    ``background_features`` is a polars frame of training features used for the
    background means that linear SHAP needs.
    """
    from sklearn.pipeline import Pipeline

    from src.explainability.attribution import unwrap

    est = unwrap(model)
    if not isinstance(est, Pipeline):
        raise TypeError(
            f"portable export supports the linear pipeline champion, got {type(est).__name__}"
        )
    prep = est.named_steps["prep"]
    clf = est.named_steps["clf"]

    # Recover the Platt parameters from the fitted calibrator.
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if not calibrated:
        raise TypeError("portable export requires a calibrated model")
    calibrator = calibrated[0].calibrators[0]
    a, b = float(calibrator.a_), float(calibrator.b_)

    def decision_of(rows: list[dict]) -> np.ndarray:
        X = _probe_frame(rows, encoder)
        return np.asarray(prep.transform(X)) @ clf.coef_.reshape(-1) + float(clf.intercept_[0])

    baseline = {name: 0.0 for name in encoder.numeric}
    baseline.update({col: BASELINE_CATEGORY for col in encoder.categorical})
    rows = [dict(baseline)]
    numeric = list(encoder.numeric)
    categorical = list(encoder.categorical)
    vocab = {col: list(encoder.vocab.get(col, [])) for col in categorical}

    # One probe row per numeric feature (unit step) and per categorical value.
    numeric_index: dict[str, int] = {}
    for name in numeric:
        numeric_index[name] = len(rows)
        rows.append({**baseline, name: 1.0})

    categorical_index: dict[str, dict[str, int]] = {}
    for col in categorical:
        categorical_index[col] = {}
        for value in vocab[col]:
            categorical_index[col][value] = len(rows)
            rows.append({**baseline, col: value})

    values = decision_of(rows)
    d0 = float(values[0])

    coefficients = {name: float(values[idx] - d0) for name, idx in numeric_index.items()}
    categorical_coefficients = {
        col: {value: float(values[idx] - d0) for value, idx in categorical_index[col].items()}
        for col in categorical
    }

    # Background means in raw feature space (for linear SHAP) and in one-hot space.
    bg = background_features
    background_means = {
        name: float(bg[name].cast(float).fill_null(encoder.fill_values.get(name, 0.0)).mean() or 0.0)
        for name in numeric
    }
    for col in categorical:
        series = bg[col].cast(str).fill_null(BASELINE_CATEGORY)
        total = max(series.len(), 1)
        for value in vocab[col]:
            share = float((series == value).sum()) / total
            background_means[f"{col}::{value}"] = share

    spec = {
        "model_version": model_version,
        "trained_at": trained_at,
        "feature_names": list(encoder.feature_names),
        "numeric": numeric,
        "categorical": categorical,
        "fill_values": encoder.fill_values,
        "vocab": vocab,
        "intercept": d0,
        "coefficients": coefficients,
        "categorical_coefficients": categorical_coefficients,
        "calibration": {"a": a, "b": b},
        "background_means": background_means,
        "reference": reference or {},
        "thresholds": thresholds,
        "probe_rows": len(rows),
    }
    log.info("portable spec exported from %d probe rows", len(rows))
    return spec
