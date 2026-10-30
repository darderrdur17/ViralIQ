"""Reference distributions that make a raw score interpretable.

A calibrated probability is not enough for a user. Two derived quantities matter:

* **Percentile** — where this score sits against a real reference population.
* **Expected view range** — what videos in the same score band actually achieved.

Both are learned from validation outcomes at training time and stored alongside
the model. Nothing here is estimated from the training set, because training-set
outcomes are what the model already memorised.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.logging import get_logger

log = get_logger("reference")

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def build_reference_bundle(scores: np.ndarray, views: np.ndarray, n_bins: int = 20) -> dict:
    """Fit score quantiles, view quantiles and per-band view outcomes.

    ``view_quantiles`` describes the view distribution of the whole validation
    population. It is what allows a prediction to be reported as a *performance*
    percentile ("this video's expected reach sits at the Nth percentile of videos")
    rather than a position within the model's own score distribution — which is
    misleading, because scores cluster tightly and tiny differences map to large
    percentile jumps.
    """
    scores = np.asarray(scores, dtype=float)
    views = np.asarray(views, dtype=float)
    if len(scores) == 0:
        raise ValueError("empty reference population")

    score_quantiles = {f"q{int(q * 100):02d}": float(np.quantile(scores, q)) for q in QUANTILES}
    view_quantiles = {
        f"q{int(q * 100):02d}": float(np.quantile(views, q))
        for q in [i / 100 for i in range(1, 100)]
    }

    # Equal-frequency bands over the score distribution.
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.array([scores.min(), np.median(scores), scores.max() + 1e-9])

    bands = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores >= lo) & (scores < hi if i < len(edges) - 2 else scores <= hi)
        if not mask.any():
            continue
        v = views[mask]
        bands.append(
            {
                "score_lower": round(float(lo), 6),
                "score_upper": round(float(hi), 6),
                "n": int(mask.sum()),
                "mean_score": round(float(scores[mask].mean()), 6),
                "views_p10": float(np.quantile(v, 0.10)),
                "views_p50": float(np.quantile(v, 0.50)),
                "views_p90": float(np.quantile(v, 0.90)),
            }
        )

    log.info(
        "reference bundle: %d score bands, %d view quantiles from %d validation rows",
        len(bands), len(view_quantiles), len(scores),
    )
    return {
        "score_quantiles": score_quantiles,
        "view_quantiles": view_quantiles,
        "bands": bands,
        "n_reference": int(len(scores)),
    }


def percentile_of_views(expected_views: float, bundle: dict) -> float | None:
    """Percentile of an expected view count within the validation view distribution."""
    q = bundle.get("view_quantiles") or {}
    if not q or expected_views is None:
        return None
    points = sorted((float(v), int(k[1:])) for k, v in q.items())
    if expected_views <= points[0][0]:
        return float(points[0][1])
    if expected_views >= points[-1][0]:
        return 100.0
    for (v0, p0), (v1, p1) in zip(points, points[1:], strict=False):
        if v0 <= expected_views <= v1:
            span = v1 - v0
            frac = 0.0 if span <= 0 else (expected_views - v0) / span
            return float(p0 + frac * (p1 - p0))
    return None


def percentile_of(score: float, bundle: dict) -> float:
    """Position of a score within the reference score distribution (0-100)."""
    q = bundle.get("score_quantiles") or {}
    if not q:
        return float("nan")
    points = sorted((float(v), int(k[1:])) for k, v in q.items())
    if score <= points[0][0]:
        return float(max(0.0, points[0][1] * score / max(points[0][0], 1e-9)))
    if score >= points[-1][0]:
        return 100.0
    for (v0, p0), (v1, p1) in zip(points, points[1:], strict=False):
        if v0 <= score <= v1:
            span = v1 - v0
            frac = 0.0 if span <= 0 else (score - v0) / span
            return float(p0 + frac * (p1 - p0))
    return 50.0


def view_range_for(score: float, bundle: dict) -> dict | None:
    """Expected view range for a score, interpolated between empirical band medians.

    Snapping to the single band containing the score would return the same estimate
    for every video in that band, destroying resolution. Piecewise-linear
    interpolation across band centres keeps the estimate monotone in the score and
    responsive to small differences.
    """
    bands = bundle.get("bands") or []
    if not bands:
        return None

    pts = sorted(bands, key=lambda b: b["mean_score"])
    xs = [b["mean_score"] for b in pts]

    def interp(key: str) -> float:
        ys = [b[key] for b in pts]
        if score <= xs[0]:
            return ys[0]
        if score >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= score <= xs[i + 1]:
                span = xs[i + 1] - xs[i]
                frac = 0.0 if span <= 0 else (score - xs[i]) / span
                return ys[i] + frac * (ys[i + 1] - ys[i])
        return ys[-1]

    nearest = min(pts, key=lambda b: abs(b["mean_score"] - score))
    return {
        "p10": round(interp("views_p10")),
        "p50": round(interp("views_p50")),
        "p90": round(interp("views_p90")),
        "band_n": nearest["n"],
        "basis": "interpolated_between_validation_score_band_medians",
    }


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")


def load_bundle(path: Path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
