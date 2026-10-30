"""Build a local, reproducible sample from remote monthly partitions.

Sampling is deterministic (hash of ``video_id``) so the same fraction always
yields the same rows, which keeps experiments reproducible without storing
fingerprints of every run.

Each partition is written to its own file. Writing per partition keeps memory
bounded, makes progress visible, and lets an interrupted build resume instead of
restarting. Downstream code reads the whole directory, so no concatenation step
is needed.
"""
from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import SETTINGS
from src.common.logging import get_logger
from src.ingestion import duckdb_engine as db
from src.ingestion.hf_source import HFDatasetSource, Partition

log = get_logger("sample")

# Columns needed for pre-publication modelling plus quality auditing.
SAMPLE_COLUMNS: tuple[str, ...] = (
    "video_id",
    "posted_at",
    "country",
    "language",
    '"type"',
    "duration_ms",
    "image_count",
    "caption",
    "hashtags",
    "sound_id",
    "is_ad",
    "branded_content",
    "is_shop_video",
    "is_ai_generated",
    "not_recommended_for_fyp",
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "stats_updated_at",
)


@dataclasses.dataclass
class SampleResult:
    name: str
    paths: list[str]
    rows: int
    months: list[str]
    fraction: float
    source_rows_scanned: int
    elapsed_seconds: float
    built_at: str
    per_partition: dict[str, int]

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def build_sample(
    months: list[str],
    fraction: float = 0.01,
    name: str | None = None,
    columns: tuple[str, ...] = SAMPLE_COLUMNS,
    force: bool = False,
) -> SampleResult:
    """Extract a deterministic sample of the given partitions to local Parquet."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    source = HFDatasetSource()
    partitions: list[Partition] = source.partitions_for(months=months)
    if not partitions:
        raise ValueError(f"No partitions matched months={months}")

    bucket = max(1, round(fraction * 100))
    name = name or "sample"
    out_dir = SETTINGS.paths.data_samples
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    scanned = 0
    per_partition: dict[str, int] = {}
    paths: list[str] = []
    col_sql = ", ".join(columns)

    with db.session() as con:
        for part in partitions:
            url = source.config.resolve_url(part.path)
            out_path = out_dir / f"{name}__{part.month}.parquet"

            n_source = con.execute(f"SELECT count(*) FROM read_parquet('{url}')").fetchone()[0]
            scanned += n_source

            if out_path.exists() and out_path.stat().st_size > 0 and not force:
                existing = con.execute(
                    f"SELECT count(*) FROM read_parquet('{out_path}')"
                ).fetchone()[0]
                log.info("%s already sampled (%d rows), skipping", part.month, existing)
                per_partition[part.month] = existing
                paths.append(str(out_path))
                continue

            t0 = time.time()
            tmp_path = out_path.with_suffix(".parquet.tmp")
            con.execute(
                f"""
                COPY (
                    SELECT {col_sql}
                    FROM read_parquet('{url}')
                    WHERE (hash(video_id) % 100) < {bucket}
                ) TO '{tmp_path}' (FORMAT PARQUET)
                """
            )
            rows = con.execute(f"SELECT count(*) FROM read_parquet('{tmp_path}')").fetchone()[0]
            tmp_path.replace(out_path)  # atomic: readers never see a partial file
            per_partition[part.month] = rows
            paths.append(str(out_path))
            log.info(
                "%s: %d of %d rows (%.2f%%) in %.1fs -> %s",
                part.month, rows, n_source, bucket, time.time() - t0, out_path.name,
            )

    result = SampleResult(
        name=name,
        paths=paths,
        rows=sum(per_partition.values()),
        months=[p.month for p in partitions],
        fraction=fraction,
        source_rows_scanned=scanned,
        elapsed_seconds=round(time.time() - started, 1),
        built_at=datetime.now(timezone.utc).isoformat(),
        per_partition=per_partition,
    )
    (out_dir / f"{name}.meta.json").write_text(
        json.dumps(result.as_dict(), indent=2), encoding="utf-8"
    )
    log.info("sample complete: %d rows across %d files", result.rows, len(paths))

    # Record the ingestion so the freshness service reflects how the corpus was
    # actually built, rather than only knowing about manual imports.
    if paths:
        try:
            import polars as pl

            from src.monitoring.freshness import record_ingestion

            summary = (
                pl.scan_parquet(paths)
                .select(["posted_at", "stats_updated_at"])
                .collect()
            )
            record_ingestion(
                rows=result.rows,
                latest_posted_at=str(summary["posted_at"].max()) if summary.height else None,
                latest_stats_updated_at=(
                    str(summary["stats_updated_at"].max()) if summary.height else None
                ),
                source=f"huggingface:{','.join(result.months)}",
                extra={
                    "sample": name,
                    "fraction": fraction,
                    "files": [Path(p).name for p in paths],
                },
            )
        except Exception as exc:  # pragma: no cover - freshness is advisory
            log.warning("could not record ingestion metadata: %s", exc)

    return result
