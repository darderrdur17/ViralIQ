"""Model registry and evaluation endpoints."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from src.common.config import SETTINGS

router = APIRouter(prefix="/models", tags=["models"])

REGISTRY = SETTINGS.paths.model_registry / "index.json"


def _registry() -> dict:
    if not REGISTRY.exists():
        raise HTTPException(
            status_code=503,
            detail="No model registry yet. Run `python scripts/train.py` to create one.",
        )
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


@router.get("")
def list_models() -> dict:
    """All training runs, the active model, and its headline metrics."""
    registry = _registry()
    runs = []
    for run in registry.get("runs", []):
        run_dir = SETTINGS.paths.model_registry / run["run_id"]
        summary = {"run_id": run["run_id"], "started_at": run.get("started_at"),
                   "champion": run.get("champion")}
        run_file = run_dir / "run.json"
        if run_file.exists():
            payload = json.loads(run_file.read_text(encoding="utf-8"))
            champion = run.get("champion")
            champion_metrics = (payload.get("metrics") or {}).get(champion) or {}
            test = champion_metrics.get("test") or {}
            summary.update(
                {
                    "splits": payload.get("splits"),
                    "pr_auc": test.get("pr_auc"),
                    "prevalence": test.get("prevalence"),
                    "sample_files": payload.get("sample_files"),
                }
            )
        runs.append(summary)

    return {
        "active_run": registry.get("active_run"),
        "active_model": registry.get("active_model"),
        "run_count": len(runs),
        "runs": runs,
    }


@router.get("/evaluation")
def model_evaluation(run_id: str | None = None) -> dict:
    """Side-by-side evaluation of every model in a run, on identical splits."""
    registry = _registry()
    run_id = run_id or registry.get("active_run")
    run_dir = SETTINGS.paths.model_registry / run_id
    run_file = run_dir / "run.json"
    if not run_file.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    payload = json.loads(run_file.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}

    comparison = []
    for name, entry in metrics.items():
        test = entry.get("test") or {}
        recent = entry.get("recent") or {}
        comparison.append(
            {
                "model": name,
                "calibrated": entry.get("calibrated"),
                "test": {
                    "n": test.get("n"),
                    "prevalence": test.get("prevalence"),
                    "pr_auc": test.get("pr_auc"),
                    "roc_auc": test.get("roc_auc"),
                    "brier": test.get("brier"),
                    "ece_10": test.get("ece_10"),
                    "ranking": test.get("ranking"),
                },
                "test_pr_auc_ci": entry.get("test_pr_auc_ci"),
                "recent": {
                    "n": recent.get("n"),
                    "prevalence": recent.get("prevalence"),
                    "pr_auc": recent.get("pr_auc"),
                    "ranking": recent.get("ranking"),
                } if recent else None,
                "note": entry.get("note"),
                "error": entry.get("error"),
            }
        )

    comparison.sort(key=lambda c: (c["test"]["pr_auc"] is None, -(c["test"]["pr_auc"] or 0)))

    # The champion is chosen on PR-AUC, but Precision@K is the operational metric a
    # content team actually acts on. When a different model wins there, say so rather
    # than letting the UI imply the champion is best at everything.
    selection_note = _selection_note(comparison, registry.get("active_model"))

    return {
        "run_id": run_id,
        "started_at": payload.get("started_at"),
        "duration_seconds": payload.get("duration_seconds"),
        "sample_files": payload.get("sample_files"),
        "splits": payload.get("splits"),
        "label_definition": payload.get("thresholds"),
        "label_diagnostics": payload.get("label_diagnostics"),
        "champion": registry.get("active_model"),
        "selection_rule": (
            "Champion is the highest test PR-AUC, breaking ties on Precision@K. PR-AUC is used "
            "because it summarises performance across the whole ranking rather than at one cutoff."
        ),
        "selection_note": selection_note,
        "comparison": comparison,
        "feature_importance": payload.get("feature_importance"),
        "environment": payload.get("environment"),
        "protocol_note": (
            "Thresholds and encoders are fitted on the training month. The test split is the "
            "second half of the following month. The recent split is a later, untouched month."
        ),
    }


def _selection_note(comparison: list[dict], champion: str | None) -> str | None:
    """Flag when the champion is not the best model at the operating cutoff."""
    if not champion:
        return None
    key = None
    for row in comparison:
        ranking = (row.get("test") or {}).get("ranking") or {}
        if ranking:
            key = next(iter(ranking))
            break
    if key is None:
        return None

    def precision(row: dict) -> float:
        return ((row.get("test") or {}).get("ranking") or {}).get(key, {}).get("precision_at_k") or 0.0

    champion_row = next((r for r in comparison if r["model"] == champion), None)
    if champion_row is None:
        return None
    best = max(comparison, key=precision)
    if best["model"] == champion or precision(best) <= precision(champion_row):
        return None
    k = key.split("=")[-1]
    return (
        f"{best['model']} has higher Precision@{k} "
        f"({precision(best) * 100:.2f}% vs {precision(champion_row) * 100:.2f}%) even though "
        f"{champion} wins on PR-AUC. If you act on a fixed-size shortlist, {best['model']} may be "
        "the better operational choice."
    )
