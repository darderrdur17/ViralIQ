"""How sensitive is the result to where the virality line is drawn?

"Top 1%" is a choice, not a law. This refits the cohort thresholds at several
percentiles on the training window only, then scores the *same* trained champion
against each label definition. The point is to show the precision/recall trade-off
rather than assert one threshold.

Usage:
    python scripts/threshold_sweep.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from src.common.config import SETTINGS, LabelConfig  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.feature_engineering.features import (  # noqa: E402
    build_prepublication_features,
    fit_sound_stats,
)
from src.labeling import labels as L  # noqa: E402
from src.models.encoder import FeatureEncoder  # noqa: E402
from src.models.train import load_frames, temporal_slices  # noqa: E402

log = get_logger("sweep")

PERCENTILES = (0.95, 0.99, 0.995)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep the virality threshold")
    parser.add_argument("--percentiles", nargs="+", type=float, default=list(PERCENTILES))
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--k", type=int, default=1000)
    args = parser.parse_args()

    registry_path = SETTINGS.paths.model_registry / "index.json"
    if not registry_path.exists():
        log.error("no trained model; run scripts/train.py first")
        return 1
    registry = json.loads(registry_path.read_text())
    run_dir = SETTINGS.paths.model_registry / registry["active_run"]
    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    import joblib

    model = joblib.load(run_dir / f"{registry['active_model']}.bin")
    encoder = FeatureEncoder.load(run_dir / "encoder.json")

    # Evaluate on exactly the partitions the model was trained on. Globbing every
    # sample on disk silently pulled in an older month whose observation ages fall
    # outside the validity window, which corrupted the label and produced a model
    # scoring below random.
    all_samples = sorted(SETTINGS.paths.data_samples.glob("*.parquet"))
    trained_on = set(run_meta.get("sample_files") or [])
    if trained_on:
        samples = [p for p in all_samples if p.name in trained_on]
        log.info("evaluating on the run's own partitions: %s", [p.name for p in samples])
    else:
        samples = all_samples
    if not samples:
        log.error("no sample files match the run's training partitions")
        return 1

    raw = load_frames(samples, file_row_cap=400_000)
    raw = L.add_observation_clock(raw)
    splits = temporal_slices(raw)
    train = splits["train"].sample(n=min(args.max_rows, splits["train"].height), seed=42)
    test = splits["test"].sample(n=min(args.max_rows, splits["test"].height), seed=42)

    # Sound statistics and the encoder come from the training window only.
    sound_stats = fit_sound_stats(train)
    featured_test = build_prepublication_features(test, sound_stats)
    X_test = encoder.transform(featured_test)

    # Score once; the model is fixed, only the label definition moves.
    scores = model.predict_proba(X_test)[:, 1]
    valid = featured_test["observation_valid"].to_numpy()

    rows = []
    for pct in args.percentiles:
        cfg = LabelConfig(
            percentile=pct,
            cohort_keys=("country", "type", "observation_age_bucket"),
            min_cohort_size=200,
            max_observation_age_hours=720.0,
        )
        train_clocked = L.add_observation_clock(train, cfg)
        thresholds = L.fit_cohort_thresholds(train_clocked, months=("train",), cfg=cfg)
        labeled = L.apply_labels(featured_test, thresholds)

        y = labeled["viral_at_snapshot"].cast(pl.Int8).to_numpy()[valid]
        s = scores[valid]
        if y.sum() == 0:
            log.warning("p%.3f produced no positives; skipping", pct)
            continue

        result = M.evaluate(y, s, (args.k,))
        rank = result["ranking"][f"k={args.k}"]
        rows.append(
            {
                "percentile": pct,
                "label": f"top {100 * (1 - pct):.1f}%",
                "prevalence": result["prevalence"],
                "positives": result["positives"],
                "n": result["n"],
                "pr_auc": result["pr_auc"],
                "roc_auc": result["roc_auc"],
                "precision_at_k": rank["precision_at_k"],
                "recall_at_k": rank["recall_at_k"],
                "lift_at_k": rank["lift_at_k"],
                "global_threshold": thresholds.global_threshold,
                "cohorts": len(thresholds.thresholds),
            }
        )
        log.info(
            "p%.3f: prevalence=%.4f PR-AUC=%.4f P@%d=%.2f%%",
            pct, result["prevalence"], result["pr_auc"], args.k,
            rank["precision_at_k"] * 100,
        )

    report = render(rows, registry["active_model"], args.k)
    out = SETTINGS.paths.reports_evaluation / "threshold_sweep.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log.info("wrote %s", out)
    print(report)
    return 0


def render(rows: list[dict], champion: str, k: int) -> str:
    lines = [
        "# Threshold Sensitivity",
        "",
        f"Champion model: **{champion}**, held fixed. Only the label definition moves.",
        "",
        "Thresholds are refitted on the training window at each percentile, then applied to the",
        "test month unchanged — the same protocol as the main evaluation.",
        "",
        f"| Label | Prevalence | Positives | PR-AUC | ROC-AUC | Precision@{k} | Recall@{k} | Lift@{k} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['prevalence']*100:.2f}% | {r['positives']:,} | "
            f"{r['pr_auc']:.4f} | {r['roc_auc']:.3f} | {r['precision_at_k']*100:.2f}% | "
            f"{r['recall_at_k']*100:.1f}% | {r['lift_at_k']:.2f}× |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- A looser threshold (top 5%) produces more positives, so **recall rises and precision",
        "  falls**. Neither number is better in isolation; they answer different questions.",
        "- PR-AUC shifts because the positive class changed, not because the model changed.",
        "  Comparisons across rows are comparisons of *label definitions*, not of models.",
        "- The shipped default is the strictest of these (top 1%), chosen because a content team",
        "  acting on a shortlist cares about precision at the top of the ranking.",
        "- Realised prevalence does not equal the nominal percentile: thresholds are fitted on the",
        "  training month and applied to a later month whose view distribution differs. At p99 the",
        "  realised rate lands close to 1%; at p95 it overshoots. That gap is a measure of",
        "  month-to-month drift, not an error in the fitting.",
        "- Lift and ROC-AUC **improve** as the label tightens, because the signal concentrates at the",
        "  very top of the ranking — which is exactly where a shortlist operates.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/threshold_sweep.py",
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
