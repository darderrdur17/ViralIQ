"""ViralIQ API.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import analytics, data, models, predictions
from api.services import store
from src.common.config import SETTINGS
from src.common.logging import get_logger
from src.monitoring import freshness

log = get_logger("api")

START_TIME = time.time()
VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    SETTINGS.ensure_dirs()
    store.init_db()
    status = freshness.get_status()
    log.info("ViralIQ API starting | data mode=%s | delay=%s days",
             status["mode"], status["publication_delay_days"])

    # Load the active model up front. Otherwise the first prediction after a restart
    # pays the deserialisation cost, which reads to a user as a hung request.
    try:
        predictor = predictions.get_predictor()
        predictor.warm()
        log.info("model preloaded and warmed: %s", predictor.version)
    except Exception as exc:
        log.warning("no model preloaded (%s); /predictions will return 503 until trained", exc)

    yield
    log.info("ViralIQ API stopping")


app = FastAPI(
    title="ViralIQ",
    description=(
        "TikTok virality intelligence. Predictions are cohort-relative and traceable to a "
        "stored model version and dataset snapshot. The service reports its own data "
        "freshness and will not present historical observations as live data."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc), "path": request.url.path},
    )


@app.get("/api/v1/health", tags=["system"])
def health() -> dict:
    """Liveness plus the two things a caller must know before trusting a number."""
    data_status = freshness.get_status()
    return {
        "status": "ok",
        "version": VERSION,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "time": datetime.now(timezone.utc).isoformat(),
        "data_mode": data_status["mode"],
        "publication_delay_days": data_status["publication_delay_days"],
        "model": freshness.active_model_info(),
    }


@app.get("/api/v1/meta", tags=["system"])
def meta() -> dict:
    """What this service is and is not, stated plainly for clients."""
    return {
        "name": "ViralIQ",
        "version": VERSION,
        "prediction_tasks": [
            {
                "task": "prepublication",
                "available": True,
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
        "data_policy": (
            "Historical-data mode is declared whenever publication coverage lags more than "
            f"{freshness.RECENT_THRESHOLD_DAYS} days."
        ),
    }


app.include_router(predictions.router, prefix="/api/v1")
app.include_router(data.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")

# Serve the dashboard same-origin so no CORS or base-URL configuration is needed.
_DASHBOARD = SETTINGS.paths.root / "dashboard"
if _DASHBOARD.exists():
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/app", StaticFiles(directory=str(_DASHBOARD), html=True), name="dashboard")

    @app.get("/", include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse(url="/app/")
