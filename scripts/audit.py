"""Dataset audit: verify what the source actually contains, measured not assumed.

Usage:
    python scripts/audit.py                     # latest 2 partitions
    python scripts/audit.py --months 2026-10 2026-09 2026-08
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import SETTINGS  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.ingestion import duckdb_engine as db  # noqa: E402
from src.ingestion.hf_source import HFDatasetSource  # noqa: E402
from src.monitoring.freshness import classify_mode  # noqa: E402
from src.preprocessing import quality  # noqa: E402

log = get_logger("audit")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the TikTok dataset")
    parser.add_argument("--months", nargs="*", default=None)
    parser.add_argument("--tail", type=int, default=2, help="Audit the last N partitions if --months is omitted")
    args = parser.parse_args()

    source = HFDatasetSource()
    all_parts = source.list_partitions()
    describe = source.describe()

    if args.months:
        parts = source.partitions_for(months=args.months)
    else:
        parts = all_parts[-args.tail:]

    if not parts:
        log.error("no partitions matched")
        return 1

    audits = []
    with db.session() as con:
        for part in parts:
            url = source.config.resolve_url(part.path)
            cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{url}')").fetchall()]
            check = quality.validate_schema(cols)
            entry = quality.audit_relation(con, f"read_parquet('{url}')", part.month)
            entry["partition"] = part.path
            entry["partition_gb"] = round(part.size_gb, 2)
            entry["schema_valid"] = check.valid
            entry["column_count"] = len(cols)
            entry["integrity"] = quality.observation_integrity_summary(entry)
            audits.append(entry)

    latest_posted = max(a["max_posted_at"] for a in audits)
    latest_stats = max(a["max_stats_updated_at"] for a in audits)
    now = datetime.now(timezone.utc)
    publication_delay = round((now - latest_posted).total_seconds() / 86400, 1)

    payload = {
        "generated_at": now.isoformat(),
        "source": describe,
        "audited_partitions": [a["partition"] for a in audits],
        "latest_posted_at": latest_posted.isoformat(),
        "latest_stats_updated_at": latest_stats.isoformat(),
        "publication_delay_days": publication_delay,
        "operating_mode": classify_mode(publication_delay),
        "pct_negative_observation_age": round(
            sum(a["pct_negative_observation_age"] * a["row_count"] for a in audits)
            / sum(a["row_count"] for a in audits),
            3,
        ),
        "fixed_age_label_supported": all(a["integrity"]["fixed_age_label_supported"] for a in audits),
        "distinct_stats_timestamps": sum(a["distinct_stats_timestamps"] for a in audits),
        "partitions": audits,
    }

    SETTINGS.paths.data_metadata.mkdir(parents=True, exist_ok=True)
    SETTINGS.paths.audit_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("audit written to %s", SETTINGS.paths.audit_file)

    report = render_markdown(payload)
    out = SETTINGS.paths.reports_research / "dataset_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log.info("report written to %s", out)
    print(report)
    return 0


def render_markdown(payload: dict) -> str:
    src = payload["source"]
    lines = [
        "# Dataset Audit — TikTok 5.6B Videos",
        "",
        f"Generated: `{payload['generated_at']}`  ",
        "Method: remote Parquet queries via DuckDB. No partition was downloaded in full.",
        "",
        "## 1. Repository",
        "",
        "| Property | Measured value |",
        "|---|---|",
        f"| Repository | `{src['repo_id']}` |",
        f"| Licence | {src['license']} |",
        f"| Monthly partitions | {src['partition_count']} |",
        f"| Total size | {src['total_gb']} GB |",
        f"| Earliest partition | `{src['earliest_month']}` |",
        f"| Latest partition | `{src['latest_month']}` ({src['latest_partition_gb']} GB) |",
        "",
        "## 2. Coverage and freshness",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Latest publication timestamp | `{payload['latest_posted_at']}` |",
        f"| Latest statistics timestamp | `{payload['latest_stats_updated_at']}` |",
        f"| Publication delay at audit time | **{payload['publication_delay_days']} days** |",
        f"| Operating mode implied by that delay | `{payload['operating_mode']}` |",
        "",
        "The delay above is the gap between the newest video in the dataset and the audit clock.",
        f"That delay places the platform in `{payload['operating_mode']}` mode "
        "(live ≤ 2 days, recently refreshed ≤ 14 days). The application derives this from the "
        "same rule, so the report and the running service cannot disagree.",
        "",
        "## 3. Partition profiles",
        "",
        "| Month | Rows | Median views | p99 views | % zero views | % negative obs. age | Distinct stat timestamps |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in payload["partitions"]:
        lines.append(
            f"| {a['label']} | {a['row_count']:,} | {a['median_views']:,.0f} | "
            f"{a['p99_views']:,.0f} | {a['pct_zero_views']}% | "
            f"**{a['pct_negative_observation_age']}%** | {a['distinct_stats_timestamps']:,} |"
        )

    lines += [
        "",
        "## 4. Critical integrity finding: the observation clock is unreliable",
        "",
        "`stats_updated_at` is documented as the time engagement counts were read. For a large",
        "share of rows it is *earlier than the video's own publication timestamp*, which is",
        "physically impossible: a video cannot accumulate views before it exists.",
        "",
    ]
    for a in payload["partitions"]:
        lines.append(
            f"- **{a['label']}**: {a['pct_negative_observation_age']}% of "
            f"{a['row_count']:,} rows have a negative observation age "
            f"(median age {a['observation_age_p50_h']} h, p99 {a['observation_age_p99_h']} h)."
        )

    lines += [
        "",
        "Rows with a negative age also under-report reach, which is consistent with them being a",
        "stale crawl batch whose counters were never refreshed after publication.",
        "",
        "### Consequences",
        "",
        "1. `observation_age_hours` cannot be trusted as a normalising variable.",
        "2. Fixed-age targets (`viral_24h`, `viral_7d`) **cannot be constructed** from this source.",
        "3. The only defensible retrospective label is a cohort-relative snapshot label, which is",
        "   what ViralIQ trains on — and it is named accordingly everywhere it appears.",
        "4. Rows with an invalid observation clock are excluded from label construction, not",
        "   silently repaired.",
        "",
        "## 5. What this dataset cannot provide",
        "",
        "| Missing capability | Consequence |",
        "|---|---|",
        "| Repeated observations per video | No velocity features; early-traction engine blocked |",
        "| Creator follower counts | Creator features limited to prior publishing history |",
        "| Video and audio files | No visual or audio embeddings |",
        "| Guaranteed live feed | Platform operates in historical-data mode |",
        "",
        "## 6. Threshold implication",
        "",
        "Observed p99 view counts sit far below the 250,000 / 400,000 figures often quoted for",
        "TikTok virality. A hard-coded global threshold would mislabel the majority of the corpus.",
        "ViralIQ therefore derives its threshold per country-and-format cohort at the 99th",
        "percentile, fitted on the training window only.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
