"""Prediction endpoints."""
from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException, Query

from api.schemas.prediction import (
    EarlyTractionRequest,
    PredictionResponse,
    PrepublishRequest,
)
from api.services import store
from src.common.logging import get_logger
from src.inference.predictor import ModelNotAvailable, Predictor

router = APIRouter(prefix="/predictions", tags=["predictions"])
log = get_logger("api.predictions")

_PREDICTOR: Predictor | None = None
_PREDICTOR_LOCK = threading.Lock()


def get_predictor() -> Predictor:
    """Return a predictor matching the registry on disk.

    The model is reloaded when the registry pointer changes, so a retrain takes
    effect without a restart. Without this the API would advertise the new run in
    /models while still scoring with the previous model.
    """
    global _PREDICTOR
    signature = Predictor.current_signature()
    if _PREDICTOR is not None and signature is not None and _PREDICTOR.signature == signature:
        return _PREDICTOR
    with _PREDICTOR_LOCK:
        if _PREDICTOR is None or signature is None or _PREDICTOR.signature != signature:
            if _PREDICTOR is not None:
                log.info("registry changed; reloading predictor")
            _PREDICTOR = Predictor()
    return _PREDICTOR


@router.post("/prepublish", response_model=PredictionResponse)
def predict_prepublish(request: PrepublishRequest) -> PredictionResponse:
    """Score a video using only information available before publication."""
    try:
        predictor = get_predictor()
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload = request.model_dump()
    try:
        result = predictor.predict_prepublish(payload)
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("prepublish prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    prediction_id = store.save_prediction(
        task="prepublication",
        model_version=result["model"]["version"],
        payload=payload,
        response=result,
    )
    result["prediction_id"] = prediction_id
    return PredictionResponse(**result)


@router.post("/early-traction", response_model=PredictionResponse)
def predict_early_traction(request: EarlyTractionRequest) -> PredictionResponse:
    """Score a video using early engagement observed before the prediction time.

    Blocked until the observation store contains real multi-timepoint data. The
    static dataset has one snapshot per video, so velocity cannot be measured from
    it and this endpoint refuses rather than approximating.
    """
    if store.observation_count() == 0:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "insufficient_observations",
                "message": (
                    "Early-traction prediction requires at least two observations of the same "
                    "video at different ages. The observation store is empty."
                ),
                "how_to_enable": [
                    "Record observations via POST /api/v1/data/observations as videos age.",
                    "The source dataset provides a single snapshot per video, so velocity "
                    "features cannot be derived from it retrospectively.",
                ],
                "current_observations": 0,
            },
        )
    raise HTTPException(
        status_code=501,
        detail={
            "error": "model_not_trained",
            "message": (
                "Observations exist but no early-traction model has been trained on them yet."
            ),
        },
    )


@router.get("/history")
def prediction_history(
    limit: int = Query(50, ge=1, le=500),
    task: str | None = Query(None),
) -> dict:
    """Stored predictions, newest first, with the payload that produced them."""
    items = store.list_predictions(limit=limit, task=task)
    return {"count": len(items), "items": items, "stats": store.prediction_stats()}
