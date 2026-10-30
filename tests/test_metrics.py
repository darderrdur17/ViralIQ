"""Tests for ranking metrics."""
from __future__ import annotations

import numpy as np

from src.evaluation import metrics as M


def test_precision_and_recall_at_k_on_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
    assert M.precision_at_k(y, s, 2) == 1.0
    assert M.recall_at_k(y, s, 2) == 1.0
    assert M.lift_at_k(y, s, 2) == 2.5  # 1.0 precision / 0.4 prevalence


def test_precision_at_k_on_inverted_ranking():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.1, 0.2, 0.9, 0.8])
    assert M.precision_at_k(y, s, 2) == 0.0
    assert M.recall_at_k(y, s, 2) == 0.0


def test_random_ranking_lands_near_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.02).astype(int)
    s = rng.random(200_000)
    p = M.precision_at_k(y, s, 5000)
    assert abs(p - y.mean()) < 0.004


def test_calibration_error_is_zero_for_perfect_calibration():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.5] * 100)
    # Predictions of 0.5 against a 0.5 base rate are calibrated overall.
    assert M.expected_calibration_error(y, p, n_bins=5) < 0.01


def test_evaluate_reports_required_metrics():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    s = np.clip(y * 0.4 + rng.random(5000) * 0.6, 0, 1)
    result = M.evaluate(y, s, (100, 1000))
    for key in ("pr_auc", "roc_auc", "brier", "log_loss", "ece_10", "ranking", "prevalence"):
        assert key in result
    assert "k=100" in result["ranking"]


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    y = (rng.random(4000) < 0.05).astype(int)
    s = np.clip(y * 0.5 + rng.random(4000) * 0.5, 0, 1)
    clusters = rng.integers(0, 200, 4000)
    from sklearn.metrics import average_precision_score

    ci = M.cluster_bootstrap_ci(y, s, clusters, average_precision_score, iterations=60)
    assert ci["lower"] <= ci["point"] <= ci["upper"]
