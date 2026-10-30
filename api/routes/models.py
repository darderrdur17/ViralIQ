"""Model registry and evaluation endpoints."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from api.schemas.prediction import PromoteRequest
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
        summary = {
            "run_id": run["run_id"],
            "started_at": run.get("started_at"),
            "champion": run.get("champion"),
            # Runs created before `status` existed fall back to the pointer rather than
            # showing "unknown" for the model that is actually serving.
            "status": run.get("status")
            or ("active" if run["run_id"] == registry.get("active_run") else "archived"),
            "is_active": run["run_id"] == registry.get("active_run"),
        }
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
        "can_promote": True,
        "candidates": registry.get("candidates", []),
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
    selection_note = _selection_note(
        comparison, registry.get("active_model"), registry.get("champion_pinned")
    )

    return {
        "run_id": run_id,
        "started_at": payload.get("started_at"),
        "duration_seconds": payload.get("duration_seconds"),
        "sample_files": payload.get("sample_files"),
        "splits": payload.get("splits"),
        "label_definition": payload.get("thresholds"),
        "label_diagnostics": payload.get("label_diagnostics"),
        "sound_coverage": payload.get("sound_coverage"),
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


def _selection_note(
    comparison: list[dict], champion: str | None, pinned: dict | None = None
) -> str | None:
    """Explain any case where the champion is not the obvious best model.

    Two situations produce a note: the champion was deliberately pinned (the demo
    mirrors the deployed linear configuration, which the fixture's own winner may not
    be), or another model wins at the operating cutoff despite losing on PR-AUC.
    Silently showing a champion that contradicts the benchmark would be the exact
    failure this project is built to avoid.
    """
    if not champion:
        return None

    champion_row = next((r for r in comparison if r["model"] == champion), None)
    if champion_row is None:
        return None

    champion_pr = (champion_row.get("test") or {}).get("pr_auc")
    pr_leader = max(
        (r for r in comparison if (r.get("test") or {}).get("pr_auc") is not None),
        key=lambda r: r["test"]["pr_auc"],
        default=None,
    )
    if pr_leader and pr_leader["model"] != champion and champion_pr is not None:
        if pr_leader["test"]["pr_auc"] > champion_pr:
            reason = (
                " This run's champion was pinned deliberately."
                if pinned and pinned.get("model") == champion
                else ""
            )
            detail = f" {pinned.get('reason')}." if pinned and pinned.get("reason") else ""
            return (
                f"{pr_leader['model']} scores higher on PR-AUC "
                f"({pr_leader['test']['pr_auc']:.4f} vs {champion_pr:.4f}) but {champion} is "
                f"reported as champion.{reason}{detail}"
            )

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


@router.post("/promote")
def promote_run(request: PromoteRequest) -> dict:
    """Make a trained run serve traffic.

    This is the human gate. Training can run unattended, but promoting a model changes
    what every user sees, and "better on PR-AUC" is a judgement that needs a person —
    a new month can move the metric more than the model difference does.
    """
    registry = _registry()
    entry = next((r for r in registry.get("runs", []) if r["run_id"] == request.run_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {request.run_id}")

    run_dir = SETTINGS.paths.model_registry / request.run_id
    run_file = run_dir / "run.json"
    if not run_file.exists():
        raise HTTPException(status_code=404, detail=f"Run artefacts missing: {request.run_id}")
    payload = json.loads(run_file.read_text(encoding="utf-8"))

    quality = payload.get("label_quality") or {}
    if quality.get("degraded"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "degraded_label",
                "message": (
                    "This run's training month lost too many rows to the observation-age "
                    "window; its threshold is not trustworthy."
                ),
                "reason": quality.get("reason"),
            },
        )

    metrics = payload.get("metrics") or {}
    eligible = {
        name: e
        for name, e in metrics.items()
        if not name.startswith("E0") and (e.get("test") or {}).get("pr_auc") is not None
    }
    if not eligible:
        raise HTTPException(status_code=422, detail="Run has no eligible model to promote")

    champion = request.model_name or max(
        eligible, key=lambda n: eligible[n]["test"]["pr_auc"]
    )
    if champion not in eligible:
        raise HTTPException(
            status_code=422,
            detail=f"{champion} is not an eligible model in this run: {sorted(eligible)}",
        )
    if not (run_dir / f"{champion}.bin").exists():
        raise HTTPException(status_code=409, detail=f"Model artefact missing: {champion}.bin")

    # The serverless build exports an affine surrogate, so a non-linear champion cannot
    # be deployed. Warn rather than block: the full service handles it fine.
    warnings: list[str] = []
    importance = (payload.get("feature_importance") or {}).get(champion) or []
    method = importance[0].get("method") if importance else None
    if method and method != "abs_coefficient":
        warnings.append(
            f"{champion} is not a linear model, so the serverless portable export will "
            "refuse to build until a linear champion is promoted."
        )
    coverage = (payload.get("sound_coverage") or {}).get("test") or {}
    if (coverage.get("pct_unseen_sound") or 0) > 50:
        warnings.append(
            f"{coverage['pct_unseen_sound']:.1f}% of this run's evaluation month uses an "
            "unseen sound, so sound features are largely absent on forward rows."
        )

    previous = {"run_id": registry.get("active_run"), "model": registry.get("active_model")}
    registry["active_run"] = request.run_id
    registry["active_model"] = champion
    registry.pop("champion_pinned", None)
    for r in registry.get("runs", []):
        r["status"] = "active" if r["run_id"] == request.run_id else "archived"
    if request.run_id in registry.get("candidates", []):
        registry["candidates"].remove(request.run_id)
    REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    return {
        "promoted": {"run_id": request.run_id, "model_name": champion},
        "previous": previous,
        "test_pr_auc": eligible[champion]["test"].get("pr_auc"),
        "warnings": warnings,
        "note": (
            "The predictor reloads on the next request. Redeploy the serverless build to "
            "reflect this change there."
        ),
    }
