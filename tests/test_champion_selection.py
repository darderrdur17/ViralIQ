"""Baselines and heuristics must never be promoted to champion.

They exist to be beaten. Before this guard, a heuristic that outscored the models on
PR-AUC would have been selected and shipped as the production model.
"""
from __future__ import annotations

from src.evaluation.report import _champion
from src.models.train import select_champion


def _entry(pr_auc: float) -> dict:
    return {"test": {"pr_auc": pr_auc}}


def test_heuristic_outscoring_models_is_not_champion():
    results = {
        "E0_random_ranking": _entry(0.01),
        "E0b_sound_popularity": _entry(0.99),  # deliberately absurd
        "E1_logistic_regression": _entry(0.03),
    }
    assert select_champion(results) == "E1_logistic_regression"


def test_champion_is_best_model_when_heuristics_lose():
    results = {
        "E0b_sound_popularity": _entry(0.015),
        "E1_logistic_regression": _entry(0.031),
        "E3_catboost": _entry(0.025),
    }
    assert select_champion(results) == "E1_logistic_regression"


def test_report_champion_ignores_baselines():
    run = {
        "metrics": {
            "E0_random_ranking": _entry(0.01),
            "E0b_sound_popularity": _entry(0.99),
            "E1_logistic_regression": _entry(0.03),
        }
    }
    assert _champion(run) == "E1_logistic_regression"


def test_only_baselines_returns_none():
    results = {"E0_random_ranking": _entry(0.01), "E0b_sound_popularity": _entry(0.02)}
    assert select_champion(results) is None
