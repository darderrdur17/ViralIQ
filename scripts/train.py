"""Train, evaluate and report.

Usage:
    python scripts/train.py
    python scripts/train.py --tag primary --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import SETTINGS  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.evaluation.report import write_reports  # noqa: E402
from src.models.train import run_training  # noqa: E402

log = get_logger("train.cli")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ViralIQ models")
    parser.add_argument("--tag", default="primary")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="Restrict training to these YYYY-MM partitions, "
             "e.g. --months 2026-09 2026-10",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Proceed even if the training month falls outside the observation-age "
             "window (produces an untrustworthy threshold).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=300_000,
        help="Per-split row cap. Bounds memory on single-machine runs; "
             "300k rows yields ~3k positives at a 1%% prevalence.",
    )
    args = parser.parse_args()

    samples = sorted(SETTINGS.paths.data_samples.glob("*.parquet"))
    if args.months:
        # Sample files are one-per-month, so a name match is enough to scope the split.
        samples = [p for p in samples if any(m in p.name for m in args.months)]
        log.info("restricted to months %s -> %s", args.months, [p.name for p in samples])
    if not samples:
        log.error("No sample data. Run scripts/build_sample.py first.")
        return 1
    log.info("samples: %s", [p.name for p in samples])

    try:
        run = run_training(
            sample_paths=samples,
            output_tag=args.tag,
            seed=args.seed,
            max_rows_per_split=args.max_rows,
            allow_degraded=args.allow_degraded,
        )
    except Exception as exc:
        if type(exc).__name__ == "DegradedLabelError":
            log.error("%s", exc)
            return 2
        raise
    run_path = SETTINGS.paths.model_registry / run.run_id / "run.json"
    reports = write_reports(run_path)

    print("\n" + "=" * 70)
    print(f"RUN {run.run_id}  ({run.duration_seconds}s)")
    print("=" * 70)
    for name, entry in sorted(
        run.metrics.items(),
        key=lambda kv: -((kv[1].get("test") or {}).get("pr_auc") or -1),
    ):
        test = entry.get("test") or {}
        pr = test.get("pr_auc")
        prev = test.get("prevalence")
        if pr is None:
            print(f"  {name:28s} no PR-AUC (error: {entry.get('error', 'n/a')})")
            continue
        print(f"  {name:28s} PR-AUC={pr:.4f}  prevalence={prev:.4f}  ROC-AUC={test.get('roc_auc')}")
    print("-" * 70)
    for key, path in reports.items():
        print(f"  {key:12s} -> {path}")
    print(json.dumps({"run_id": run.run_id, "reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
