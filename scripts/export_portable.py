"""Export everything the serverless deployment needs.

The deployed site runs without scikit-learn, CatBoost, LightGBM, pandas or DuckDB, so
two things must be produced locally and committed:

1. a portable model spec (see ``src/inference/portable.py``) plus sound statistics;
2. pre-computed API responses for the read-only analytics routes, because the 341 MB
   sample corpus cannot be shipped to a serverless function.

Usage:
    python scripts/export_portable.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from api.services import analytics  # noqa: E402
from src.common.config import SETTINGS  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.inference.portable import export_portable_spec  # noqa: E402
from src.inference.predictor import Predictor  # noqa: E402
from src.monitoring import freshness, health  # noqa: E402

log = get_logger("export")

DEFAULT_OUT = SETTINGS.paths.root / "deploy" / "vercel" / "data"


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s (%.1f KB)", path.name, path.stat().st_size / 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export portable model + static API payloads")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out: Path = args.out

    predictor = Predictor()
    background = pl.read_parquet(
        predictor.run_dir / "reference_features.parquet"
    ).select(predictor.encoder.feature_names)

    spec = export_portable_spec(
        predictor.model,
        predictor.encoder,
        predictor.reference,
        predictor.thresholds,
        background,
        predictor.version,
        predictor.run_meta.get("started_at"),
    )
    _dump(out / "portable_model.json", spec)

    if predictor.sound_stats is not None:
        out.mkdir(parents=True, exist_ok=True)
        predictor.sound_stats.write_parquet(out / "sound_stats.parquet")
        log.info(
            "wrote sound_stats.parquet (%.1f MB)",
            (out / "sound_stats.parquet").stat().st_size / 1e6,
        )

    # ---- pre-computed read-only API payloads -----------------------------
    api_dir = out / "api"
    _dump(api_dir / "overview.json", analytics.overview())
    _dump(api_dir / "content.json", analytics.content_analytics())

    trends = analytics.trends(limit=12)
    if trends.get("available"):
        trends["mode"] = freshness.get_status(check_source=False)["mode"]
        trends["caveat"] = (
            "Ranked from historical observations of the ingested sample. These are not live "
            "TikTok Creative Center trends."
        )
    _dump(api_dir / "trends.json", trends)

    status = freshness.get_status(check_source=False)
    status["model"] = freshness.active_model_info()
    status["sample"] = {
        "available": analytics.is_ready(),
        "files": [p.name for p in sorted(SETTINGS.paths.data_samples.glob("*.parquet"))],
    }
    try:
        status["health"] = health.model_health()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("health assessment failed: %s", exc)
        status["health"] = {"verdict": "unknown", "reasons": [str(exc)]}

    status["deployment"] = {
        "kind": "serverless",
        "note": (
            "Read-only analytics are pre-computed at export time. Prediction runs live "
            "through a portable model; there is no persistent store in this deployment."
        ),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    _dump(api_dir / "status.json", status)

    registry_path = SETTINGS.paths.model_registry / "index.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    run_id = registry["active_run"]
    run = json.loads(
        (SETTINGS.paths.model_registry / run_id / "run.json").read_text(encoding="utf-8")
    )

    runs = []
    for entry in registry.get("runs", []):
        runs.append(
            {
                "run_id": entry["run_id"],
                "started_at": entry.get("started_at"),
                "champion": entry.get("champion"),
                "pr_auc": ((run.get("metrics", {}).get(entry.get("champion")) or {}).get("test") or {}).get("pr_auc"),
                "splits": run.get("splits"),
                "sample_files": run.get("sample_files"),
            }
        )
    _dump(
        api_dir / "models.json",
        {
            "active_run": run_id,
            "active_model": registry.get("active_model"),
            "run_count": len(runs),
            "runs": runs,
        },
    )

    # Reuse the live route's own comparison logic so the deployed view cannot drift.
    from api.routes.models import _selection_note

    comparison = []
    for name, entry in (run.get("metrics") or {}).items():
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
                }
                if recent
                else None,
                "note": entry.get("note"),
                "error": entry.get("error"),
            }
        )
    comparison.sort(key=lambda c: (c["test"]["pr_auc"] is None, -(c["test"]["pr_auc"] or 0)))
    _dump(
        api_dir / "evaluation.json",
        {
            "run_id": run_id,
            "started_at": run.get("started_at"),
            "duration_seconds": run.get("duration_seconds"),
            "sample_files": run.get("sample_files"),
            "splits": run.get("splits"),
            "label_definition": run.get("thresholds"),
            "label_diagnostics": run.get("label_diagnostics"),
            "sound_coverage": run.get("sound_coverage"),
            "champion": registry.get("active_model"),
            "selection_rule": (
                "Champion is the highest test PR-AUC, breaking ties on Precision@K. PR-AUC is "
                "used because it summarises performance across the whole ranking."
            ),
            "selection_note": _selection_note(comparison, registry.get("active_model")),
            "comparison": comparison,
            "feature_importance": run.get("feature_importance"),
            "environment": run.get("environment"),
            "protocol_note": (
                "Thresholds and encoders are fitted on the training month. The test split is the "
                "second half of the following month."
            ),
        },
    )

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log.info("export complete: %.1f MB in %s", total / 1e6, out)

    _sync_runtime(out.parent)
    return 0


def _sync_runtime(project: Path) -> None:
    """Mirror the modules the function imports, and the dashboard.

    Vercel builds the ``deploy/vercel`` directory as a standalone project, so the
    packages it imports must exist inside it. The mirror is generated, never edited
    by hand, and ``tests/test_portable.py`` asserts the deployed inference path still
    matches the real predictor.
    """
    import shutil

    source_root = SETTINGS.paths.root
    src_src = source_root / "src"
    src_dst = project / "src"
    if src_dst.exists():
        shutil.rmtree(src_dst)
    shutil.copytree(
        src_src,
        src_dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    log.info("mirrored src/ -> %s", src_dst)

    public = project / "public"
    public.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "dashboard" / "index.html", public / "index.html")
    log.info("copied dashboard -> %s", public / "index.html")


if __name__ == "__main__":
    raise SystemExit(main())
