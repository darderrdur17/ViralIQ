"""Data freshness monitoring.

The platform must never imply it holds live TikTok statistics when it does not.
This service converts measured timestamps into an explicit operating mode:
``live``, ``recently_refreshed``, ``historical``, or ``unavailable``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import SETTINGS
from src.common.logging import get_logger
from src.ingestion.hf_source import HFDatasetSource

log = get_logger("freshness")

# A trend-sensitive platform cannot call two-week-old data current. These bounds are
# deliberately tight: TikTok sound and hashtag cycles move on a weekly cadence, so a
# feed that lags by more than a fortnight is historical for analytical purposes even
# if the repository itself was uploaded yesterday.
LIVE_THRESHOLD_DAYS = 2
RECENT_THRESHOLD_DAYS = 14


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def classify_mode(delay_days: float | None) -> str:
    if delay_days is None:
        return "unavailable"
    if delay_days <= LIVE_THRESHOLD_DAYS:
        return "live"
    if delay_days <= RECENT_THRESHOLD_DAYS:
        return "recently_refreshed"
    return "historical"


def read_audit() -> dict:
    path = SETTINGS.paths.audit_file
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def record_ingestion(
    *,
    rows: int,
    latest_posted_at: str | None,
    latest_stats_updated_at: str | None,
    source: str,
    extra: dict | None = None,
) -> dict:
    """Persist the outcome of an ingestion run. This is the audit trail."""
    payload = {
        "last_ingestion_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "rows_ingested": rows,
        "latest_posted_at": latest_posted_at,
        "latest_stats_updated_at": latest_stats_updated_at,
        **(extra or {}),
    }
    SETTINGS.paths.data_metadata.mkdir(parents=True, exist_ok=True)
    SETTINGS.paths.freshness_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("ingestion recorded: %s rows from %s", rows, source)
    return payload


def get_status(check_source: bool = False) -> dict:
    """Full freshness report for the dashboard and API."""
    audit = read_audit()
    recorded = {}
    if SETTINGS.paths.freshness_file.exists():
        try:
            recorded = json.loads(SETTINGS.paths.freshness_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            recorded = {}

    latest_posted = (
        audit.get("latest_posted_at")
        or recorded.get("latest_posted_at")
    )
    latest_stats = (
        audit.get("latest_stats_updated_at")
        or recorded.get("latest_stats_updated_at")
    )

    now = datetime.now(timezone.utc)
    posted_dt = _parse(latest_posted)
    stats_dt = _parse(latest_stats)

    publication_delay_days = round((now - posted_dt).total_seconds() / 86400, 1) if posted_dt else None
    observation_delay_days = round((now - stats_dt).total_seconds() / 86400, 1) if stats_dt else None
    mode = classify_mode(publication_delay_days)

    source_status: dict = {"name": "huggingface", "checked": False}
    if check_source:
        src = HFDatasetSource()
        ok, err = src.is_available()
        source_status = {"name": "huggingface", "checked": True, "available": ok, "error": err}
        if ok:
            desc = src.describe()
            source_status.update(
                {
                    "repo_id": desc["repo_id"],
                    "license": desc["license"],
                    "partition_count": desc["partition_count"],
                    "total_gb": desc["total_gb"],
                    "latest_partition": desc["latest_month"],
                }
            )
    else:
        # Serve the last audited description so the dashboard has real source facts
        # without making the caller wait on a network round trip.
        audit_source = audit.get("source") or {}
        if audit_source:
            source_status.update(
                {
                    "checked": False,
                    "from_audit": True,
                    "repo_id": audit_source.get("repo_id"),
                    "license": audit_source.get("license"),
                    "partition_count": audit_source.get("partition_count"),
                    "total_gb": audit_source.get("total_gb"),
                    "latest_partition": audit_source.get("latest_month"),
                    "audited_at": audit_source.get("checked_at"),
                }
            )

    reasons = {
        "live": f"Publication coverage is within {LIVE_THRESHOLD_DAYS} days of today.",
        "recently_refreshed": (
            f"Publication coverage is {LIVE_THRESHOLD_DAYS}-{RECENT_THRESHOLD_DAYS} days old."
        ),
        "historical": (
            f"Latest publication in the dataset is {publication_delay_days} days old. "
            f"This platform is operating in historical-data mode and does not hold live "
            f"TikTok statistics."
        ),
        "unavailable": "No ingestion has been recorded yet.",
    }

    return {
        "mode": mode,
        "mode_explanation": reasons[mode],
        "is_live": mode == "live",
        "latest_posted_at": latest_posted,
        "latest_stats_updated_at": latest_stats,
        "publication_delay_days": publication_delay_days,
        "observation_delay_days": observation_delay_days,
        "last_ingestion_at": recorded.get("last_ingestion_at"),
        "rows_ingested": recorded.get("rows_ingested"),
        "thresholds_days": {
            "live": LIVE_THRESHOLD_DAYS,
            "recently_refreshed": RECENT_THRESHOLD_DAYS,
        },
        "source": source_status,
        "data_quality": {
            "pct_negative_observation_age": audit.get("pct_negative_observation_age"),
            "fixed_age_label_supported": audit.get("fixed_age_label_supported"),
            "distinct_stats_timestamps": audit.get("distinct_stats_timestamps"),
        },
        "checked_at": now.isoformat(),
    }


def active_model_info() -> dict:
    """Current model pointer plus last retraining time."""
    registry_path: Path = SETTINGS.paths.model_registry / "index.json"
    if not registry_path.exists():
        return {"available": False, "reason": "no_registry"}
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    run_id = registry.get("active_run")
    run_dir = SETTINGS.paths.model_registry / run_id if run_id else None
    started_at = None
    if run_dir and (run_dir / "run.json").exists():
        started_at = json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("started_at")
    return {
        "available": bool(run_id),
        "run_id": run_id,
        "model_name": registry.get("active_model"),
        "trained_at": started_at,
        "run_count": len(registry.get("runs", [])),
    }
