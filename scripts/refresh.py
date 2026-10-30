"""Refresh the platform when new data lands.

Replaces the manual three-step dance (pick months by hand, build samples, train) with
one command that decides which months are usable from measured observation ages, and
tells you what changed.

Usage:
    python scripts/refresh.py --dry-run        # plan only, no writes
    python scripts/refresh.py --check-source   # also ask Hugging Face for the newest partition
    python scripts/refresh.py --fetch          # download samples for months not on disk
    python scripts/refresh.py                  # retrain on usable months, then export
    python scripts/refresh.py --check-only     # exit 2 if degraded, for cron/monitoring
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import SETTINGS  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.monitoring import health  # noqa: E402

log = get_logger("refresh")

EXIT_OK = 0
EXIT_STALE = 1
EXIT_DEGRADED = 2


def _latest_remote_partition() -> str | None:
    """Newest monthly partition the source currently publishes. One API call."""
    try:
        from src.ingestion.hf_source import HFDatasetSource

        parts = HFDatasetSource().list_partitions(refresh=True)
        return parts[-1].month if parts else None
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("could not reach the source: %s", exc)
        return None


def _print_plan(state: dict, remote_latest: str | None) -> None:
    print()
    print("=" * 72)
    print("ViralIQ refresh plan")
    print("=" * 72)

    months = state["months"]
    if not months:
        print("  No local samples. Nothing can be trained.")
    else:
        print(f"  {'month':9s} {'rows':>10s} {'median age':>12s} {'>window':>9s}  status")
        for m in months:
            flag = "usable" if m["usable"] else "REJECTED"
            print(
                f"  {m['month']:9s} {m['rows']:>10,} {m['median_age_hours']:>10,.0f}h "
                f"{m['pct_over_window']:>8.1f}%  {flag}"
            )
            if not m["usable"]:
                print(f"            {m['reason']}")

    if remote_latest:
        local = {m["month"] for m in months}
        if remote_latest not in local:
            print(f"\n  New upstream data: partition {remote_latest} is published but not local.")
            print("  Fetch it with: python scripts/refresh.py --fetch")
        else:
            print(f"\n  Upstream latest partition ({remote_latest}) is already local.")

    print(f"\n  Train on: {state['usable_months'] or 'nothing usable'}")
    if state["rejected_months"]:
        print(f"  Skipped:  {state['rejected_months']} (observation ages too old)")

    model = state.get("model") or {}
    if model.get("run_id"):
        pr = model.get("champion_metrics")
        print(f"\n  Active model: {model.get('model_name')} @ {model['run_id']}")
        print(f"    champion test PR-AUC: {pr if pr is not None else '—'}")
    else:
        print("\n  Active model: none")

    print(f"\n  Verdict: {state['verdict'].upper()}")
    for reason in state["reasons"]:
        print(f"    - {reason}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh data, model and reports")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no writes")
    parser.add_argument("--check-only", action="store_true", help="Report and exit; no writes")
    parser.add_argument("--check-source", action="store_true", help="Query the source for new partitions")
    parser.add_argument("--fetch", action="store_true", help="Build samples for months not on disk")
    parser.add_argument("--fraction", type=float, default=0.01, help="Sampling fraction for fetched months")
    parser.add_argument("--max-rows", type=int, default=300_000, help="Per-split row cap when training")
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Train and measure a candidate without promoting it; the active model keeps serving.",
    )
    args = parser.parse_args()

    state = health.model_health()
    remote_latest = _latest_remote_partition() if args.check_source else None
    _print_plan(state, remote_latest)

    if args.dry_run or args.check_only:
        if state["verdict"] == "degraded":
            return EXIT_DEGRADED
        if state["verdict"] == "no_model" or state["verdict"] == "current_with_caveats":
            return EXIT_STALE
        return EXIT_OK

    if len(state["usable_months"]) < 2:
        log.error(
            "need at least two usable months; found %s. Fetch data first "
            "(python scripts/refresh.py --fetch).",
            state["usable_months"],
        )
        return EXIT_DEGRADED

    # Optionally pull months that are published upstream but absent locally.
    if args.fetch:
        from src.ingestion.sample_builder import build_sample

        local = {m["month"] for m in state["months"]}
        candidates = [remote_latest] if remote_latest and remote_latest not in local else []
        for month in candidates:
            log.info("fetching %s at %.0f%%", month, args.fraction * 100)
            build_sample(months=[month], fraction=args.fraction, name=f"refresh_{month}")
        if candidates:
            state = health.model_health()

    selected = state["usable_months"]

    # Compare against the run being replaced, so the report can say what moved.
    previous_pr_auc = (state.get("model") or {}).get("champion_metrics")

    from src.evaluation.report import write_reports
    from src.models.train import run_training

    # Scope to the months just selected. Globbing the sample directory would pull in
    # rejected months and trip the label-quality guard — the very footgun this script
    # exists to remove.
    sample_paths = [
        p
        for p in sorted(SETTINGS.paths.data_samples.glob("*.parquet"))
        if any(month in p.name for month in selected)
    ]
    if not sample_paths:
        log.error("no sample files match the selected months %s", selected)
        return EXIT_DEGRADED

    log.info("training on %s", selected)
    run = run_training(
        sample_paths=sample_paths,
        output_tag="candidate" if args.candidate_only else "refresh",
        max_rows_per_split=args.max_rows,
        activate=not args.candidate_only,
    )
    # Published reports describe the model that is serving. A candidate must never
    # overwrite them: doing so silently contradicts the README and the dashboard, which
    # still point at the active run.
    if args.candidate_only:
        log.info("candidate run %s: published reports left untouched", run.run_id)
    else:
        write_reports(SETTINGS.paths.model_registry / run.run_id / "run.json")

    champion = max(
        (
            ((entry.get("test") or {}).get("pr_auc") or -1, name)
            for name, entry in run.metrics.items()
            if not name.startswith("E0")
        ),
        default=(None, None),
    )[1]
    new_pr_auc = ((run.metrics.get(champion) or {}).get("test") or {}).get("pr_auc")
    coverage = (run.sound_coverage or {}).get("test") or {}

    print()
    print("=" * 72)
    print("Refresh complete")
    print("=" * 72)
    print(f"  run        {run.run_id}")
    print(f"  months     {', '.join(selected)}")
    print(f"  champion   {champion}")
    if new_pr_auc is not None:
        if previous_pr_auc is not None:
            delta = new_pr_auc - float(previous_pr_auc)
            print(f"  PR-AUC     {new_pr_auc:.4f}  (was {float(previous_pr_auc):.4f}, {delta:+.4f})")
        else:
            print(f"  PR-AUC     {new_pr_auc:.4f}  (no previous run to compare)")
    print(f"  sound cov. {coverage.get('pct_unseen_sound', '—')}% of test rows use an unseen sound")
    if args.candidate_only:
        print()
        print("  Stored as a CANDIDATE. The active model still serves traffic.")
        print("  Promote it from the dashboard (Model Laboratory) or:")
        print("    curl -X POST localhost:8000/api/v1/models/promote -H 'Content-Type: application/json' \\")
        print(f"         -d '{{\"run_id\": \"{run.run_id}\"}}'")
    print()
    print("  Next: redeploy the serverless build if you ship one")
    print("        python scripts/export_portable.py && (cd deploy/vercel && vercel --prod)")
    print("=" * 72)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
