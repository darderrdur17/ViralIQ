"""Ranking-first evaluation metrics.

Viral videos are rare, so accuracy is meaningless here. The primary question is:
*among the videos we ranked highest, how many actually went viral, and how much
better is that than picking at random?*

Every metric returns both a point estimate and, where useful, a clustered
bootstrap interval. Clustering matters because rows are not independent — videos
sharing a sound or a crawl batch move together.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    if y.shape != s.shape:
        raise ValueError("y_true and y_score must have the same length")
    return y, s


def precision_at_k(y_true, y_score, k: int) -> float:
    """Share of the top-k ranked items that are positive."""
    y, s = _as_arrays(y_true, y_score)
    k = min(k, len(y))
    if k == 0:
        return float("nan")
    idx = np.argsort(-s, kind="stable")[:k]
    return float(y[idx].mean())


def recall_at_k(y_true, y_score, k: int) -> float:
    """Share of all positives captured inside the top-k."""
    y, s = _as_arrays(y_true, y_score)
    k = min(k, len(y))
    total_pos = int(y.sum())
    if total_pos == 0 or k == 0:
        return float("nan")
    idx = np.argsort(-s, kind="stable")[:k]
    return float(y[idx].sum() / total_pos)


def lift_at_k(y_true, y_score, k: int) -> float:
    """Precision@k divided by the base rate — the multiple over random picking."""
    y, _ = _as_arrays(y_true, y_score)
    base = float(y.mean())
    if base == 0:
        return float("nan")
    return float(precision_at_k(y, y_score, k) / base)


def _bin_edges(probs: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Bin edges for calibration analysis.

    Equal-width bins are useless for rare-event probabilities: with a ~1% base rate
    every prediction falls in the first bin, producing a one-row calibration table.
    Equal-frequency (quantile) bins are the default for that reason.
    """
    if strategy == "quantile":
        edges = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 3:
            return np.array([probs.min(), float(np.median(probs)), probs.max() + 1e-12])
        edges[0] = min(edges[0], probs.min())
        edges[-1] = max(edges[-1], probs.max())
        return edges
    return np.linspace(0.0, 1.0, n_bins + 1)


def expected_calibration_error(
    y_true, y_prob, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    """Binned gap between predicted probability and observed frequency."""
    y, p = _as_arrays(y_true, y_prob)
    edges = _bin_edges(p, n_bins, strategy)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / len(y)) * abs(acc - conf)
    return float(ece)


def calibration_table(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile") -> list[dict]:
    y, p = _as_arrays(y_true, y_prob)
    edges = _bin_edges(p, n_bins, strategy)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": round(float(edges[b]), 6),
                "bin_upper": round(float(edges[b + 1]), 6),
                "count": int(mask.sum()),
                "mean_predicted": round(float(p[mask].mean()), 6),
                "observed_rate": round(float(y[mask].mean()), 6),
            }
        )
    return rows


def evaluate(y_true, y_score, k_values: tuple[int, ...], threshold: float = 0.5) -> dict:
    """Full metric bundle for one model on one evaluation population."""
    y, s = _as_arrays(y_true, y_score)
    prevalence = float(y.mean())
    pred = (s >= threshold).astype(int)

    result: dict = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": round(prevalence, 6),
        "pr_auc": round(float(average_precision_score(y, s)), 6),
        "roc_auc": round(float(roc_auc_score(y, s)), 6) if 0 < y.sum() < len(y) else None,
        "brier": round(float(brier_score_loss(y, np.clip(s, 0, 1))), 6),
        "log_loss": round(float(log_loss(y, np.clip(s, 1e-9, 1 - 1e-9), labels=[0, 1])), 6),
        "f1_at_0_5": round(float(f1_score(y, pred, zero_division=0)), 6),
        "ece_10": round(expected_calibration_error(y, s), 6),
        "ranking": {},
        "calibration": calibration_table(y, s),
    }
    for k in k_values:
        if k > len(y):
            continue
        result["ranking"][f"k={k}"] = {
            "precision_at_k": round(precision_at_k(y, s, k), 6),
            "recall_at_k": round(recall_at_k(y, s, k), 6),
            "lift_at_k": round(lift_at_k(y, s, k), 4),
        }
    return result


def cluster_bootstrap_ci(
    y_true,
    y_score,
    cluster_ids,
    metric_fn,
    iterations: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap that resamples whole clusters.

    Rows sharing a cluster (e.g. a sound, or a crawl batch) are correlated, so
    resampling individual rows would understate uncertainty.
    """
    y, s = _as_arrays(y_true, y_score)
    clusters = np.asarray(cluster_ids)
    unique = np.unique(clusters)
    if len(unique) < 2:
        return {"point": float(metric_fn(y, s)), "lower": None, "upper": None, "iterations": 0}

    rng = np.random.default_rng(seed)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
    samples: list[float] = []
    for _ in range(iterations):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in picked])
        yy, ss = y[idx], s[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        try:
            samples.append(float(metric_fn(yy, ss)))
        except ValueError:
            continue

    if not samples:
        return {"point": float(metric_fn(y, s)), "lower": None, "upper": None, "iterations": 0}
    arr = np.asarray(samples)
    return {
        "point": round(float(metric_fn(y, s)), 6),
        "lower": round(float(np.quantile(arr, alpha / 2)), 6),
        "upper": round(float(np.quantile(arr, 1 - alpha / 2)), 6),
        "iterations": int(len(arr)),
    }


def random_ranking_baseline(y_true, k_values: tuple[int, ...], seed: int = 42) -> dict:
    """The floor every model must beat: random scores."""
    y = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    return evaluate(y, rng.random(len(y)), k_values)
