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


def test_note_explains_a_pinned_champion():
    """A pinned champion must not silently contradict the benchmark.

    The demo pins E1 because the serverless export needs an affine model. If the
    fixture's natural winner is a forest, the Model Laboratory has to say so rather
    than showing a champion that the table contradicts.
    """
    from api.routes.models import _selection_note

    comparison = [
        {"model": "E2_random_forest", "test": {"pr_auc": 0.0080, "ranking": {"k=1000": {"precision_at_k": 0.02}}}},
        {"model": "E1_logistic_regression", "test": {"pr_auc": 0.0030, "ranking": {"k=1000": {"precision_at_k": 0.01}}}},
    ]
    pinned = {"model": "E1_logistic_regression", "reason": "pinned to the deployed linear champion"}

    note = _selection_note(comparison, "E1_logistic_regression", pinned)
    assert note is not None
    assert "E2_random_forest scores higher on PR-AUC" in note
    assert "pinned deliberately" in note
    assert "deployed linear champion" in note


def test_no_note_when_champion_actually_leads():
    from api.routes.models import _selection_note

    comparison = [
        {"model": "E1_logistic_regression", "test": {"pr_auc": 0.0318, "ranking": {"k=1000": {"precision_at_k": 0.025}}}},
        {"model": "E3_catboost", "test": {"pr_auc": 0.0250, "ranking": {"k=1000": {"precision_at_k": 0.037}}}},
    ]
    # Champion leads on PR-AUC but loses at the cutoff: the existing note should fire.
    note = _selection_note(comparison, "E1_logistic_regression")
    assert note is not None and "Precision@1000" in note

    # Champion leads on both: silence.
    comparison[1]["test"]["ranking"]["k=1000"]["precision_at_k"] = 0.01
    assert _selection_note(comparison, "E1_logistic_regression") is None
