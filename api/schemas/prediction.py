"""Request/response contracts for the inference API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PrepublishRequest(BaseModel):
    """Metadata that genuinely exists before a video is published.

    There is deliberately no field for views, likes, or any other realised
    engagement metric. ``extra="forbid"`` makes that a hard contract: a caller who
    sends ``views`` gets a validation error rather than having the field silently
    dropped while believing it influenced the score.
    """

    model_config = ConfigDict(extra="forbid")

    caption: str = Field("", description="Video caption text", max_length=4000)
    hashtags: list[str] = Field(default_factory=list, description="Hashtags without the # prefix")
    duration_ms: int = Field(0, ge=0, le=3_600_000)
    country: str | None = Field(None, max_length=8, description="ISO-ish country code as TikTok reports it")
    language: str | None = Field(None, max_length=16)
    type: str = Field("video", description="Content type as reported by the platform", max_length=32)
    posted_at: datetime | None = Field(None, description="Intended publication time; defaults to now (UTC)")
    sound_id: int | None = Field(None, ge=0, le=2**63 - 1, description="Sound identifier, if already chosen")
    video_id: int | None = Field(None, ge=0, le=2**63 - 1)
    image_count: int = Field(0, ge=0, le=100)
    is_ad: bool = False
    is_shop_video: bool = False
    is_ai_generated: bool = False
    not_recommended_for_fyp: bool = False
    branded_content: str | None = Field(None, max_length=64)


class EarlyTractionRequest(BaseModel):
    """Features observed after publication, before the prediction timestamp."""

    model_config = ConfigDict(extra="forbid")

    video_id: int = Field(..., ge=0)
    views: int = Field(..., ge=0)
    likes: int = Field(0, ge=0)
    comments: int = Field(0, ge=0)
    shares: int = Field(0, ge=0)
    saves: int = Field(0, ge=0)
    observation_age_hours: float = Field(..., gt=0, description="Hours between publication and observation")


class PredictionResponse(BaseModel):
    prediction_id: str | None = None
    task: str
    viral_probability: float
    virality_tier: str
    tier_description: str
    predicted_percentile: float | None = None
    percentile_basis: str | None = None
    expected_views: dict[str, Any] | None = None
    decision_threshold_views: int | None = None
    label_definition: str | None = None
    input_warnings: list[str] = Field(default_factory=list)
    drivers: dict[str, Any] | None = None
    model: dict[str, Any] | None = None
    inputs_used: dict[str, Any] | None = None
    generated_at: str | None = None
    limitations: list[str] = Field(default_factory=list)


class ImportRequest(BaseModel):
    path: str = Field(..., description="Absolute path to a local .parquet or .csv file")
    source_label: str = Field("manual_import")
    copy_to_raw: bool = True


class PromoteRequest(BaseModel):
    """Promote a trained run to serve traffic."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., description="Run to promote, e.g. from GET /models")
    model_name: str | None = Field(
        None, description="Override the champion; defaults to the run's best PR-AUC model"
    )
