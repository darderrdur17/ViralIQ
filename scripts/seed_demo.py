"""Seed a working demo from the committed fixture.

A fresh clone has no data and no model, so `uvicorn` alone renders an empty dashboard
and every prediction returns 503. This script makes the project demo-able in about
half a minute, with no network access and no multi-gigabyte download.

The fixture is a real 20,000-row sample of the source dataset (October 2026), not
synthetic data — so the numbers on screen are real measurements of real videos, just
from a small slice. See `data/fixtures/README.md` for provenance and licence.

Usage:
    python scripts/seed_demo.py
    uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from src.common.config import SETTINGS  # noqa: E402
from src.common.logging import get_logger  # noqa: E402

log = get_logger("seed")

FIXTURE = SETTINGS.paths.root / "data" / "fixtures" / "demo_sample.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed a demo dataset and model")
    parser.add_argument("--rows", type=int, default=20_000, help="Training row cap")
    parser.add_argument(
        "--champion",
        default="E1_logistic_regression",
        help="Model to pin as active. Defaults to the linear champion the platform "
             "actually deploys, so the demo exercises the same inference path.",
    )
    parser.add_argument("--force", action="store_true", help="Re-seed over an existing demo")
    args = parser.parse_args()

    if not FIXTURE.exists():
        log.error("fixture missing: %s", FIXTURE)
        return 1

    target = SETTINGS.paths.data_samples / "demo.parquet"
    if target.exists() and not args.force:
        log.info("demo sample already present (%s); use --force to rebuild", target.name)
    else:
        shutil.copy2(FIXTURE, target)
        log.info("copied fixture -> %s", target)

    # Provenance record, so sampling weights and freshness reporting stay coherent.
    meta = {
        "name": "demo",
        "paths": [str(target)],
        "rows": pl.read_parquet(target).height,
        "months": ["2026-10"],
        "fraction": 1.0,
        "source_rows_scanned": None,
        "elapsed_seconds": 0.0,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "per_partition": {"2026-10": pl.read_parquet(target).height},
        "note": "seeded from the committed fixture by scripts/seed_demo.py",
    }
    (SETTINGS.paths.data_samples / "demo.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Record the ingestion, otherwise the header badge says "unavailable data" while
    # the dashboard is clearly showing data.
    from src.monitoring.freshness import record_ingestion

    fixture_df = pl.read_parquet(target)
    record_ingestion(
        rows=fixture_df.height,
        latest_posted_at=str(fixture_df["posted_at"].max()),
        latest_stats_updated_at=str(fixture_df["stats_updated_at"].max()),
        source="demo_fixture",
        extra={"sample": "demo", "note": "seeded by scripts/seed_demo.py"},
    )

    from src.evaluation.report import write_reports
    from src.models.train import run_training

    log.info("training on the fixture (this takes a few seconds)")
    # Scope to the fixture only: other sample files may exist on a developer machine
    # and must not leak into the demo model.
    run = run_training(
        sample_paths=[target],
        output_tag="demo",
        max_rows_per_split=args.rows,
        file_row_cap=args.rows,
    )
    write_reports(SETTINGS.paths.model_registry / run.run_id / "run.json")

    # Pin the champion. The demo exists to mirror the product, and the product ships a
    # linear model because the serverless deployment exports an affine surrogate of it.
    # On a 20k slice a forest can win on PR-AUC, which would leave the demo — and CI's
    # parity tests — exercising a path the real deployment never uses.
    registry_path = SETTINGS.paths.model_registry / "index.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    fixture_winner = registry.get("active_model")
    pinned = args.champion
    if pinned and pinned != fixture_winner:
        if not (SETTINGS.paths.model_registry / run.run_id / f"{pinned}.bin").exists():
            log.warning("cannot pin %s: it was not trained in this run", pinned)
        else:
            registry["active_model"] = pinned
            # Record why, so the Model Laboratory can explain a champion that does not
            # top the PR-AUC table rather than quietly contradicting it.
            registry["champion_pinned"] = {
                "model": pinned,
                "fixture_winner": fixture_winner,
                "reason": (
                    "This demo is pinned to the linear champion the platform actually "
                    "deploys, because the serverless export requires an affine model"
                ),
            }
            registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
            log.info(
                "pinned champion %s (fixture winner was %s) so the demo matches the "
                "deployed linear configuration",
                pinned,
                fixture_winner,
            )

    elif "champion_pinned" in registry:
        registry.pop("champion_pinned", None)
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    active_model = json.loads(registry_path.read_text())["active_model"]

    print()
    print("=" * 68)
    print("Demo seeded.")
    print(f"  run      {run.run_id}")
    print(f"  rows     {run.splits.get('train', {}).get('rows', 0):,} train")
    print(f"  champion {active_model}")
    if pinned and pinned != fixture_winner:
        print(f"           (pinned; the fixture's own PR-AUC winner was {fixture_winner})")
    print()
    print("  uvicorn api.main:app --port 8000")
    print("  open    http://localhost:8000/app/")
    print("=" * 68)
    print()
    print("This is a 20k-row slice. For the real evaluation, build a full sample:")
    print("  python scripts/build_sample.py --months 2026-08 2026-09 --fraction 0.01")
    print("  python scripts/build_sample.py --months 2026-10 --fraction 1.0 --name r")
    print("  python scripts/train.py --months 2026-09 2026-10 --max-rows 300000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
