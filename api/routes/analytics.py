"""Analytics and trend endpoints. All figures are live SQL aggregates."""
from __future__ import annotations

from fastapi import APIRouter, Query

from api.services import analytics
from src.monitoring import freshness

router = APIRouter(tags=["analytics"])


@router.get("/analytics/overview")
def overview() -> dict:
    """Dataset scale, coverage, distribution and observed virality prevalence."""
    data = analytics.overview()
    data["freshness"] = freshness.get_status(check_source=False)
    if data.get("available"):
        data["prevalence"] = analytics.virality_prevalence()
    return data


@router.get("/analytics/content")
def content() -> dict:
    """Duration, publication-hour and country performance breakdowns."""
    return analytics.content_analytics()


@router.get("/trends")
def trends(limit: int = Query(15, ge=1, le=50)) -> dict:
    """Accelerating hashtags and sounds within the observed window."""
    data = analytics.trends(limit=limit)
    if data.get("available"):
        data["mode"] = freshness.get_status(check_source=False)["mode"]
        data["caveat"] = (
            "Ranked from historical observations of the ingested sample. These are not live "
            "TikTok Creative Center trends."
        )
    return data
