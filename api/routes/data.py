"""Data operations endpoints: freshness, schema, ingestion."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas.prediction import ImportRequest
from api.services import analytics, store
from src.common.config import SETTINGS
from src.common.logging import get_logger
from src.ingestion.hf_source import HFDatasetSource
from src.ingestion.importer import ImportRejected, load_and_normalise, store_sample
from src.monitoring import freshness
from src.preprocessing import quality

router = APIRouter(prefix="/data", tags=["data"])
log = get_logger("api.data")

ALLOWED_SUFFIXES = {".parquet", ".csv"}


@router.get("/status")
def data_status(check_source: bool = False) -> dict:
    """Freshness, quality and model-recency in one payload.

    ``mode`` is authoritative: when it reads ``historical`` the dashboard must not
    present any figure as live TikTok data.
    """
    status = freshness.get_status(check_source=check_source)
    status["model"] = freshness.active_model_info()
    status["sample"] = {
        "available": analytics.is_ready(),
        "files": [p.name for p in sorted(SETTINGS.paths.data_samples.glob("*.parquet"))],
    }
    return status


@router.get("/schema")
def data_schema(check_source: bool = True) -> dict:
    """Verify the live source schema against the expected contract."""
    if not check_source:
        raise HTTPException(status_code=400, detail="schema check requires check_source=true")
    from src.ingestion import duckdb_engine as db

    src = HFDatasetSource()
    parts = src.partitions_for()
    if not parts:
        raise HTTPException(status_code=503, detail="No partitions available")
    url = src.config.resolve_url(parts[-1].path)
    with db.session() as con:
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{url}')").fetchall()]
    check = quality.validate_schema(cols)
    return {
        "available": True,
        "partition": parts[-1].path,
        "columns": cols,
        "check": check.as_dict(),
    }


def _ingest_path(path: Path, source_label: str) -> dict:
    """Validate, normalise and persist an external file as a sample."""
    try:
        df, result = load_and_normalise(path)
        stored = store_sample(df, path, SETTINGS.paths.data_samples)
    except ImportRejected as exc:
        # 422 for content problems, 404 when the path simply is not there.
        status = 404 if "File not found" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    latest_posted = df["posted_at"].max() if "posted_at" in df.columns else None
    latest_stats = df["stats_updated_at"].max() if "stats_updated_at" in df.columns else None

    record = freshness.record_ingestion(
        rows=df.height,
        latest_posted_at=str(latest_posted) if latest_posted else None,
        latest_stats_updated_at=str(latest_stats) if latest_stats else None,
        source=source_label,
        extra={
            "file": str(stored),
            "original": result.path,
            "original_format": result.original_format,
            "normalisations": result.normalisations,
            "columns": result.columns,
        },
    )
    return {
        "status": "ingested",
        "rows": result.rows,
        "columns": result.columns,
        "stored_at": str(stored),
        "original_format": result.original_format,
        "normalisations": result.normalisations,
        "schema_check": quality.validate_schema(list(df.columns)).as_dict(),
        "ingestion": record,
    }


@router.post("/import")
def import_file(request: ImportRequest) -> dict:
    """Import a local Parquet or CSV file into the sample store."""
    return _ingest_path(Path(request.path).expanduser(), request.source_label)


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    source_label: str = "upload",
) -> dict:
    """Upload a Parquet or CSV file directly."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix or '(no extension)'}. Use .parquet or .csv",
        )

    target = SETTINGS.paths.data_raw / (file.filename or f"upload{suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return _ingest_path(target, source_label)


@router.get("/observations")
def list_observations(limit: int = 100) -> dict:
    """Recorded multi-timepoint observations (empty until collection begins)."""
    return {
        "count": store.observation_count(),
        "limit": limit,
        "note": (
            "Observations are the foundation of genuine forecasting. Populate this table by "
            "re-crawling videos at 30m / 1h / 6h / 24h / 7d after publication."
        ),
    }
