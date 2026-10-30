"""ViralIQ — serverless deployment (Vercel).

This is the same product, packaged for a 250 MB serverless limit. The full stack
(scikit-learn + SciPy + pandas + CatBoost + LightGBM + DuckDB ≈ 478 MB) cannot fit, so:

* **Prediction runs live**, through a portable affine surrogate of the champion model
  exported from the real fitted pipeline (`scripts/export_portable.py`). Parity with the
  real predictor is asserted in `tests/test_portable.py` at 1e-9.
* **Analytics are pre-computed** at export time, because the 341 MB sample corpus cannot
  ship to a function. Every payload carries the timestamp it was measured at.
* **Nothing is persisted.** Prediction history is empty here and says so.

What this deployment will not do: pretend it is the full service. `/data/import` and
`/predictions/history` are honest about being unavailable.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering.features import build_prepublication_features  # noqa: E402
from src.inference.portable import PortableModel  # noqa: E402

DATA = ROOT / "data"
API_DIR = DATA / "api"
PUBLIC = ROOT / "public"

VERSION = "1.0.0"
STARTED = datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Asset loading (once per cold start)
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_model() -> PortableModel:
    stats_path = DATA / "sound_stats.parquet"
    return PortableModel.load(
        DATA / "portable_model.json",
        stats_path if stats_path.exists() else None,
    )


MODEL: PortableModel | None
try:
    MODEL = _load_model()
except Exception as exc:  # pragma: no cover - cold-start failure path
    MODEL = None
    print(f"[viraliq] portable model unavailable: {exc}", file=sys.stderr)

STATUS = _read_json(API_DIR / "status.json")
OVERVIEW = _read_json(API_DIR / "overview.json")
CONTENT = _read_json(API_DIR / "content.json")
TRENDS = _read_json(API_DIR / "trends.json")
MODELS = _read_json(API_DIR / "models.json")
EVALUATION = _read_json(API_DIR / "evaluation.json")


app = FastAPI(
    title="ViralIQ",
    description=(
        "TikTok virality intelligence. Predictions are cohort-relative and traceable to a "
        "stored model version. Analytics are a pre-computed snapshot; the payloads state "
        "when they were measured."
    ),
    version=VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas (mirrors api/schemas/prediction.py)
# --------------------------------------------------------------------------- #
class PrepublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caption: str = Field("", max_length=4000)
    hashtags: list[str] = Field(default_factory=list)
    duration_ms: int = Field(0, ge=0, le=3_600_000)
    country: str | None = Field(None, max_length=8)
    language: str | None = Field(None, max_length=16)
    type: str = Field("video", max_length=32)
    posted_at: datetime | None = None
    sound_id: int | None = Field(None, ge=0, le=2**63 - 1)
    video_id: int | None = Field(None, ge=0, le=2**63 - 1)
    image_count: int = Field(0, ge=0, le=100)
    is_ad: bool = False
    is_shop_video: bool = False
    is_ai_generated: bool = False
    not_recommended_for_fyp: bool = False
    branded_content: str | None = Field(None, max_length=64)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tier(percentile: float | None) -> tuple[str, str]:
    if percentile is None:
        return "unknown", "No reference distribution available to place this score."
    bands = (
        (95.0, "very_high", "Top 5% of scored videos — strong virality candidate"),
        (80.0, "high", "Above the peer-group norm"),
        (50.0, "moderate", "Around the peer-group median"),
        (20.0, "low", "Below the peer-group norm"),
        (0.0, "very_low", "Unlikely to reach peer-group top 1%"),
    )
    for threshold, label, desc in bands:
        if percentile >= threshold:
            return label, desc
    return "very_low", bands[-1][2]


def _percentile_of_views(expected: float, bundle: dict) -> float | None:
    q = bundle.get("view_quantiles") or {}
    if not q:
        return None
    points = sorted((float(v), int(k[1:])) for k, v in q.items())
    if expected <= points[0][0]:
        return float(points[0][1])
    if expected >= points[-1][0]:
        return 100.0
    for (v0, p0), (v1, p1) in zip(points, points[1:], strict=False):
        if v0 <= expected <= v1:
            span = v1 - v0
            frac = 0.0 if span <= 0 else (expected - v0) / span
            return float(p0 + frac * (p1 - p0))
    return None


def _view_range(score: float, bundle: dict) -> dict | None:
    bands = bundle.get("bands") or []
    if not bands:
        return None
    pts = sorted(bands, key=lambda b: b["mean_score"])
    xs = [b["mean_score"] for b in pts]

    def interp(key: str) -> float:
        ys = [b[key] for b in pts]
        if score <= xs[0]:
            return ys[0]
        if score >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= score <= xs[i + 1]:
                span = xs[i + 1] - xs[i]
                frac = 0.0 if span <= 0 else (score - xs[i]) / span
                return ys[i] + frac * (ys[i + 1] - ys[i])
        return ys[-1]

    nearest = min(pts, key=lambda b: abs(b["mean_score"] - score))
    return {
        "p10": round(interp("views_p10")),
        "p50": round(interp("views_p50")),
        "p90": round(interp("views_p90")),
        "band_n": nearest["n"],
        "basis": "interpolated_between_validation_score_band_medians",
    }


def _input_warnings(payload: dict) -> list[str]:
    warnings = []
    if not (payload.get("caption") or "").strip():
        warnings.append("caption was empty; caption features were defaulted")
    if not payload.get("duration_ms"):
        warnings.append("duration_ms was not supplied; duration was imputed from training data")
    if not payload.get("country"):
        warnings.append("country was not supplied; the global cohort fallback was used")
    if not payload.get("language"):
        warnings.append("language was not supplied; language was treated as unknown")
    if not payload.get("hashtags"):
        warnings.append("no hashtags were supplied")
    if not payload.get("sound_id"):
        warnings.append("no sound_id was supplied; sound features were imputed")
    return warnings


def _row(payload: dict) -> pl.DataFrame:
    posted = payload.get("posted_at") or datetime.now(timezone.utc)
    if isinstance(posted, str):
        posted = datetime.fromisoformat(posted.replace("Z", "+00:00"))
    if posted.tzinfo is None:
        posted = posted.replace(tzinfo=timezone.utc)

    hashtags = payload.get("hashtags") or []
    if isinstance(hashtags, str):
        hashtags = [h.strip().lstrip("#") for h in hashtags.split(",") if h.strip()]

    sound_id = payload.get("sound_id")
    row = {
        "video_id": int(payload.get("video_id") or 0),
        "posted_at": posted,
        "country": payload.get("country") or "__na__",
        "language": payload.get("language") or "__na__",
        "type": payload.get("type") or "video",
        "duration_ms": int(payload.get("duration_ms") or 0),
        "image_count": int(payload.get("image_count") or 0),
        "caption": payload.get("caption") or "",
        "hashtags": hashtags,
        "sound_id": int(sound_id) if sound_id is not None else None,
        "is_ad": int(bool(payload.get("is_ad"))),
        "branded_content": payload.get("branded_content") or None,
        "is_shop_video": int(bool(payload.get("is_shop_video"))),
        "is_ai_generated": int(bool(payload.get("is_ai_generated"))),
        "not_recommended_for_fyp": int(bool(payload.get("not_recommended_for_fyp"))),
    }
    return pl.DataFrame([row]).with_columns(
        pl.col("posted_at").cast(pl.Datetime("us", "UTC")),
        pl.col("sound_id").cast(pl.Int64, strict=False),
        pl.col("video_id").cast(pl.Int64, strict=False),
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/v1/health")
def health() -> dict:
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "version": VERSION,
        "deployment": "serverless",
        "uptime_seconds": round((datetime.now(timezone.utc) - STARTED).total_seconds(), 1),
        "time": datetime.now(timezone.utc).isoformat(),
        "data_mode": STATUS.get("mode", "unavailable"),
        "publication_delay_days": STATUS.get("publication_delay_days"),
        "model": {
            "available": MODEL is not None,
            "run_id": MODEL.model_version.split("@")[-1] if MODEL else None,
            "model_name": MODEL.model_version.split("@")[0] if MODEL else None,
            "trained_at": MODEL.trained_at if MODEL else None,
        },
    }


@app.get("/api/v1/meta")
def meta() -> dict:
    return {
        "name": "ViralIQ",
        "version": VERSION,
        "deployment": "serverless",
        "prediction_tasks": [
            {
                "task": "prepublication",
                "available": MODEL is not None,
                "description": "Cohort-relative top-1% virality probability from metadata only.",
            },
            {
                "task": "early_traction",
                "available": False,
                "description": "Requires repeated observations of the same video over time.",
                "blocked_by": "observation_store_empty",
            },
        ],
        "metric_policy": (
            "Ranking metrics (Precision@K, Recall@K, Lift@K, PR-AUC) are primary. Accuracy is "
            "not reported because viral videos are rare."
        ),
        "deployment_note": (
            "Analytics are a pre-computed snapshot taken at export time. Prediction runs live "
            "through a portable surrogate of the champion model. Nothing is persisted."
        ),
    }


@app.post("/api/v1/predictions/prepublish")
def predict_prepublish(request: PrepublishRequest) -> dict:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Portable model unavailable in this deployment")

    payload = request.model_dump()
    featured = build_prepublication_features(_row(payload), MODEL.sound_stats)
    features = {name: featured[name][0] for name in MODEL.feature_names}

    prob = MODEL.probability(features)
    bundle = MODEL.reference or {}
    view_range = _view_range(prob, bundle)
    percentile = (
        _percentile_of_views(view_range["p50"], bundle) if view_range else None
    )
    percentile = round(percentile, 1) if percentile is not None else None
    tier, tier_desc = _tier(percentile)

    threshold = (MODEL.thresholds or {}).get("global_threshold")
    warnings = _input_warnings(payload)
    limitations = [
        "This is a retrospective snapshot model: it estimates whether a video lands in its "
        "cohort's top 1% of observed views, not its exact view count after seven days.",
        "Creator follower counts are not present in the source dataset and are not used.",
        "Sound popularity is frozen at the training window and does not reflect today's trends.",
        "Feature attributions describe model behaviour, not causal effects on virality.",
    ]
    if warnings:
        limitations.insert(
            0,
            "This prediction was produced from an incomplete description; the fields listed in "
            "input_warnings were defaulted and the score is correspondingly less reliable.",
        )

    keys = (MODEL.thresholds or {}).get("keys") or ["country", "type"]
    readable = " × ".join(str(k).replace("_", " ") for k in keys)
    pct_label = 100 * (1 - (MODEL.thresholds or {}).get("percentile", 0.99))

    return {
        "prediction_id": None,
        "task": "prepublication",
        "viral_probability": round(prob, 4),
        "virality_tier": tier,
        "tier_description": tier_desc,
        "predicted_percentile": percentile,
        "percentile_basis": (
            "Percentile of this video's expected median reach within the observed view "
            f"distribution of the validation population (n={bundle.get('n_reference')}). "
            "Resolution is limited by the width of the score band it falls in."
        ),
        "expected_views": view_range,
        "decision_threshold_views": round(threshold) if threshold else None,
        "label_definition": (
            f"Top {pct_label:.0f}% of observed views within the same {readable} peer group, "
            "at snapshot time."
        ),
        "input_warnings": warnings,
        "drivers": {
            "method": "shap_linear_analytic",
            "top": MODEL.attributions(features),
            "note": (
                "Linear SHAP: coefficient × deviation from the training background mean. "
                "Describes the ranking signal, not a causal effect on virality."
            ),
        },
        "model": {
            "version": MODEL.model_version,
            "run_id": MODEL.model_version.split("@")[-1],
            "model_name": MODEL.model_version.split("@")[0],
            "trained_at": MODEL.trained_at,
            "deployment": "portable_surrogate",
        },
        "inputs_used": payload,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limitations": limitations,
    }


@app.post("/api/v1/predictions/early-traction")
def predict_early_traction() -> JSONResponse:
    raise HTTPException(
        status_code=409,
        detail={
            "error": "insufficient_observations",
            "message": (
                "Early-traction prediction requires at least two observations of the same video "
                "at different ages. The observation store is empty."
            ),
            "how_to_enable": [
                "Record observations as videos age (30m / 1h / 6h / 24h / 7d).",
                "The source dataset provides a single snapshot per video, so velocity cannot be "
                "derived from it retrospectively.",
            ],
        },
    )


@app.get("/api/v1/predictions/history")
def history(limit: int = Query(50, ge=1, le=500), task: str | None = None) -> dict:
    return {
        "count": 0,
        "items": [],
        "stats": {
            "total_predictions": 0,
            "mean_viral_probability": None,
            "latest_prediction_at": None,
            "observations_recorded": 0,
        },
        "deployment_note": (
            "This serverless deployment has no persistent store, so prediction history is not "
            "retained. Run the full service locally or in a container to persist predictions."
        ),
    }


@app.get("/api/v1/analytics/overview")
def analytics_overview() -> dict:
    payload = dict(OVERVIEW)
    payload["freshness"] = STATUS
    return payload


@app.get("/api/v1/analytics/content")
def analytics_content() -> dict:
    return CONTENT


@app.get("/api/v1/trends")
def trends(limit: int = Query(15, ge=1, le=50)) -> dict:
    payload = dict(TRENDS)
    if payload.get("accelerating_hashtags"):
        payload["accelerating_hashtags"] = payload["accelerating_hashtags"][:limit]
    if payload.get("accelerating_sounds"):
        payload["accelerating_sounds"] = payload["accelerating_sounds"][:limit]
    return payload


@app.get("/api/v1/models")
def models() -> dict:
    if not MODELS:
        raise HTTPException(status_code=503, detail="No model registry exported")
    return MODELS


@app.get("/api/v1/models/evaluation")
def model_evaluation(run_id: str | None = None) -> dict:
    if not EVALUATION:
        raise HTTPException(status_code=503, detail="No evaluation exported")
    if run_id and run_id != EVALUATION.get("run_id"):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return EVALUATION


@app.get("/api/v1/data/status")
def data_status(check_source: bool = False) -> dict:
    return STATUS


@app.get("/api/v1/data/schema")
def data_schema(check_source: bool = True) -> dict:
    raise HTTPException(
        status_code=503,
        detail=(
            "Schema verification queries the Hugging Face source directly and is not available "
            "in the serverless deployment. Run scripts/audit.py locally."
        ),
    )


@app.post("/api/v1/data/import")
@app.post("/api/v1/data/upload")
def data_import() -> dict:
    raise HTTPException(
        status_code=503,
        detail={
            "error": "read_only_deployment",
            "message": (
                "Ingestion writes to a local sample corpus, which a serverless function does not "
                "have. Use the container or local deployment to import data."
            ),
        },
    )


@app.get("/api/v1/data/observations")
def observations(limit: int = 100) -> dict:
    return {
        "count": 0,
        "limit": limit,
        "note": "No observation store in the serverless deployment.",
    }


# --------------------------------------------------------------------------- #
# Static dashboard
# --------------------------------------------------------------------------- #
if PUBLIC.exists():
    app.mount("/app", StaticFiles(directory=str(PUBLIC), html=True), name="dashboard")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(PUBLIC / "index.html")
