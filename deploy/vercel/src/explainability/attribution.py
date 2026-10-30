"""Feature attribution for ViralIQ models.

Two consumers share this module so that training-time global importance and
inference-time per-prediction attribution can never drift apart:

* ``global_importance`` — used by the training pipeline to populate the model
  registry for every model family, including linear ones.
* ``local_attributions`` — used by the inference service to explain a single score.

The awkward part is one-hot encoding. A pipeline model is fitted on expanded dummy
columns, so a raw SHAP value belongs to ``cat__country_US`` rather than to
``country``. Both helpers aggregate dummy contributions back onto their parent
input column, which keeps explanations readable and makes linear and tree models
report in the same vocabulary.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.feature_engineering.features import CATEGORICAL_FEATURES

log = get_logger("explainability")

TREE_MARKERS = ("get_booster", "get_feature_importance", "booster_", "feature_importances_")


def unwrap(model) -> Any:
    """Strip calibration / frozen wrappers to reach the fitted estimator."""
    est = model
    calibrated = getattr(est, "calibrated_classifiers_", None)
    if calibrated:
        est = calibrated[0].estimator
    return getattr(est, "estimator", est)


def aggregate_to_inputs(
    values: np.ndarray, expanded_names: list[str], categorical: tuple[str, ...] = CATEGORICAL_FEATURES
) -> dict[str, float]:
    """Sum dummy-column contributions onto their parent input column."""
    if len(values) != len(expanded_names):
        raise ValueError(
            f"attribution length mismatch: {len(values)} values vs {len(expanded_names)} names"
        )
    totals: dict[str, float] = {}
    cat_set = set(categorical)
    for value, name in zip(values, expanded_names, strict=True):
        body = str(name).split("__", 1)[-1]
        parent = body
        if str(name).startswith("cat__"):
            for col in cat_set:
                if body == col or body.startswith(col + "_"):
                    parent = col
                    break
        totals[parent] = totals.get(parent, 0.0) + float(value)
    return totals


def _ranked(totals: dict[str, float], top_n: int) -> list[tuple[str, float]]:
    return sorted(totals.items(), key=lambda kv: -abs(kv[1]))[:top_n]


def global_importance(model, top_n: int = 20) -> list[dict] | None:
    """Model-level feature importance, aggregated to input columns.

    Returns ``None`` only when the estimator family genuinely has no importance
    concept, so the caller can say "unavailable" instead of showing an empty chart.
    """
    from sklearn.pipeline import Pipeline

    est = unwrap(model)

    try:
        if isinstance(est, Pipeline):
            prep = est.named_steps.get("prep")
            clf = est.named_steps.get("clf")
            if prep is None or clf is None:
                return None
            names = list(prep.get_feature_names_out())

            coef = getattr(clf, "coef_", None)
            if coef is not None:
                raw = np.abs(np.asarray(coef).reshape(-1))
                method = "abs_coefficient"
            else:
                raw = np.asarray(getattr(clf, "feature_importances_", []), dtype=float)
                method = "impurity"
                if raw.size == 0:
                    return None
            totals = aggregate_to_inputs(raw, names)
        else:
            catboost_imp = getattr(est, "get_feature_importance", None)
            if callable(catboost_imp):
                raw = np.asarray(catboost_imp(), dtype=float)
                names = list(getattr(est, "feature_names_", []) or [])
                method = "catboost_prediction_value"
            else:
                booster = getattr(est, "booster_", None)
                if booster is not None and hasattr(booster, "feature_importance"):
                    raw = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
                    names = list(booster.feature_name())
                    method = "lightgbm_gain"
                else:
                    raw = np.asarray(getattr(est, "feature_importances_", []), dtype=float)
                    names = list(getattr(est, "feature_names_in_", []) or [])
                    method = "impurity"
            if raw.size == 0 or len(names) != raw.size:
                return None
            totals = {n: float(v) for n, v in zip(names, raw, strict=True)}

        total = sum(abs(v) for v in totals.values()) or 1.0
        return [
            {"feature": feat, "importance": round(abs(val) / total, 5), "method": method}
            for feat, val in _ranked(totals, top_n)
        ]
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("global importance failed: %s", exc)
        return None


def local_attributions(
    model, X, reference=None, encoder=None, top_n: int = 8, cache: dict | None = None
) -> dict | None:
    """Per-prediction attribution in input-feature space.

    ``reference``/``encoder`` supply a background distribution. Without one, a row
    explained against itself yields all-zero attributions, which looks like a bug to
    a user even though it is mathematically correct.

    ``cache`` avoids rebuilding the explainer (and transforming the background
    sample) on every request; constructing it costs seconds and it is valid for the
    lifetime of the loaded model.
    """
    import shap
    from sklearn.pipeline import Pipeline

    est = unwrap(model)
    cache = cache if cache is not None else {}
    key = ("explainer", id(est))
    explainer = cache.get(key)

    if isinstance(est, Pipeline):
        prep = est.named_steps.get("prep")
        clf = est.named_steps.get("clf")
        if prep is None or clf is None:
            return None
        Xt = prep.transform(X)
        names = list(prep.get_feature_names_out())
        if explainer is None:
            if reference is not None and encoder is not None:
                background = prep.transform(encoder.transform(reference))
                explainer = shap.LinearExplainer(
                    clf, shap.maskers.Independent(background, max_samples=background.shape[0])
                )
            else:
                explainer = shap.LinearExplainer(clf, Xt)
            cache[key] = explainer
        values = np.asarray(explainer.shap_values(Xt)).reshape(-1)
        totals = aggregate_to_inputs(values, names)
        method = "shap_linear_aggregated"
        note = (
            "One-hot contributions are summed back onto their parent input column. "
            "Attributions come from the model before probability calibration and describe "
            "the ranking signal, not a causal effect on virality."
        )
    else:
        if not any(hasattr(est, m) for m in TREE_MARKERS):
            return None
        if explainer is None:
            explainer = shap.TreeExplainer(est)
            cache[key] = explainer
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[-1]
        arr = np.asarray(values).reshape(-1)
        names = list(X.columns)
        totals = {n: float(v) for n, v in zip(names, arr, strict=False)}
        method = "shap_tree"
        note = (
            "Attributions come from the tree model before probability calibration, so they "
            "explain the ranking signal rather than the final probability scale."
        )

    ranked = _ranked(totals, top_n)
    return {
        "method": method,
        "note": note,
        "top": [
            {
                "feature": feat,
                "value": _jsonable(X.iloc[0].get(feat)),
                "contribution": round(val, 5),
                "direction": "increases" if val > 0 else "decreases",
            }
            for feat, val in ranked
        ],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
