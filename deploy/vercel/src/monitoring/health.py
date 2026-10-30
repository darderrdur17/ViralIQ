"""Operational health: which months are usable, and is the model still current?

Two questions a user actually asks when new data lands:

1. *Which months can I train on?* The answer is not "the ones on disk" — a partition
   whose rows were crawled long after publication is a biased subsample (see
   ``reports/research/three_month_drift_experiment.md``). This module decides it from
   measured observation ages rather than from the calendar.
2. *Is what I am looking at still trustworthy?* Freshness, label quality and sound
   coverage each degrade independently, so the verdict is assembled from all three.

Everything here reads local artefacts only; it never touches the network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from src.common.config import SETTINGS
from src.common.logging import get_logger

log = get_logger("health")

# A month is usable when at most this share of its rows was observed past the age
# window. Matches the training guard's threshold.
MAX_OVER_WINDOW_PCT = 25.0
# Beyond this, the platform is in historical-data mode rather than merely behind.
STALE_DATA_DAYS = 30.0
# Above this, the strongest feature family is mostly absent on forward rows.
BLIND_SOUNDS_PCT = 50.0


@dataclass(frozen=True)
class MonthHealth:
    month: str
    file: str
    rows: int
    median_age_hours: float
    pct_over_window: float
    pct_negative_age: float
    usable: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def month_health(sample_dir: Path | None = None) -> list[MonthHealth]:
    """Profile each locally available monthly sample.

    Cheap: a single DuckDB pass over the local Parquet files.
    """
    from src.ingestion import duckdb_engine as db

    directory = sample_dir or SETTINGS.paths.data_samples
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return []

    joined = ", ".join(f"'{f}'" for f in files)
    window = SETTINGS.labels.max_observation_age_hours
    sql = f"""
    WITH t AS (
        SELECT regexp_extract(filename, '(\\d{{4}}-\\d{{2}})', 1) AS month,
               (epoch(stats_updated_at) - epoch(posted_at)) / 3600.0 AS age
        FROM read_parquet([{joined}], filename=true)
        WHERE stats_updated_at IS NOT NULL
    )
    SELECT month,
           count(*)                                                        AS rows,
           round(median(age), 1)                                           AS median_age,
           round(100.0 * avg(CASE WHEN age > {window} THEN 1 ELSE 0 END), 2) AS over_window,
           round(100.0 * avg(CASE WHEN age < 0 THEN 1 ELSE 0 END), 2)      AS negative_age
    FROM t GROUP BY 1 ORDER BY 1
    """
    with db.session() as con:
        rows = con.execute(sql).fetchall()

    out: list[MonthHealth] = []
    for month, n, median_age, over, negative in rows:
        if month is None or n == 0:
            continue
        usable = over <= MAX_OVER_WINDOW_PCT
        reason = (
            "within the observation-age window"
            if usable
            else (
                f"{over:.1f}% of rows were observed more than {window:.0f}h after "
                "publication; the survivors are a biased subsample"
            )
        )
        out.append(
            MonthHealth(
                month=month,
                file=next((f.name for f in files if month in f.name), ""),
                rows=int(n),
                median_age_hours=float(median_age or 0.0),
                pct_over_window=float(over or 0.0),
                pct_negative_age=float(negative or 0.0),
                usable=usable,
                reason=reason,
            )
        )
    return out


def select_months(months: list[MonthHealth], max_months: int = 3) -> tuple[list[str], list[str]]:
    """Pick the newest usable months. Returns ``(selected, rejected)``."""
    usable = sorted((m for m in months if m.usable), key=lambda m: m.month)
    rejected = [m.month for m in sorted(months, key=lambda m: m.month) if not m.usable]
    selected = [m.month for m in usable[-max_months:]]
    return selected, rejected


def model_health() -> dict:
    """Assemble a verdict from freshness, label quality and sound coverage."""
    from src.monitoring import freshness

    reasons: list[str] = []
    status = freshness.get_status()
    months = month_health()
    selected, rejected = select_months(months)

    registry_path = SETTINGS.paths.model_registry / "index.json"
    run_meta: dict = {}
    model_name = None
    run_id = None
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        run_id = registry.get("active_run")
        model_name = registry.get("active_model")
        if run_id:
            run_file = SETTINGS.paths.model_registry / run_id / "run.json"
            if run_file.exists():
                run_meta = json.loads(run_file.read_text(encoding="utf-8"))

    if not run_meta:
        return {
            "verdict": "no_model",
            "reasons": ["No trained model. Run scripts/refresh.py or scripts/seed_demo.py."],
            "months": [m.as_dict() for m in months],
            "usable_months": selected,
            "rejected_months": rejected,
            "model": {"run_id": run_id, "model_name": model_name},
        }

    delay = status.get("publication_delay_days")
    label_quality = run_meta.get("label_quality") or {}
    coverage = (run_meta.get("sound_coverage") or {}).get("test") or {}
    unseen_pct = coverage.get("pct_unseen_sound")

    if delay is not None and delay > STALE_DATA_DAYS:
        reasons.append(
            f"Dataset publication coverage is {delay:.0f} days old, so the platform is in "
            "historical-data mode."
        )
    if label_quality.get("degraded"):
        reasons.append(
            f"Training month lost {label_quality.get('pct_over_age_window', 0):.1f}% of rows to "
            "the observation-age window."
        )
    if unseen_pct is not None and unseen_pct > BLIND_SOUNDS_PCT:
        reasons.append(
            f"{unseen_pct:.1f}% of the evaluation month uses a sound unseen in training, so "
            "sound features are largely absent on forward rows."
        )
    if len(selected) < 2:
        reasons.append(
            "Fewer than two usable months are on disk; a temporal split needs at least two."
        )

    if label_quality.get("degraded") or len(selected) < 2:
        verdict = "degraded"
    elif reasons:
        verdict = "current_with_caveats"
    else:
        verdict = "current"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "data_lag_days": delay,
        "sound_coverage_test_pct": unseen_pct,
        "label_quality": label_quality,
        "months": [m.as_dict() for m in months],
        "usable_months": selected,
        "rejected_months": rejected,
        "model": {
            "run_id": run_id,
            "model_name": model_name,
            "trained_at": run_meta.get("started_at"),
            "champion_metrics": (
                (run_meta.get("metrics", {}).get(model_name) or {}).get("test") or {}
            ).get("pr_auc"),
        },
    }
